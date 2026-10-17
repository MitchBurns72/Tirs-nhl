"""
Pipeline quotidien. Roule dans la GitHub Action (3 fois par jour) :

  1. GRADER   : les prédictions des soirs passés dont les matchs sont finis
                reçoivent leur résultat (tirs au but réels, via la feuille de match).
  2. PRÉDIRE  : pour chaque match de ce soir pas encore commencé, toutes les
                versions de l'algo calculent P(2+ tirs) pour chaque joueur.
  3. PUBLIER  : docs/data.json est réécrit; la page le lit.

Lancement : python src/run.py            (date du jour, heure de l'Est)
            python src/run.py 2026-10-18 (forcer une date, pratique pour tester)
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import nhl_api as api
from modeles import VERSIONS, VERSION_PRINCIPALE
from score import resumer

RACINE = Path(__file__).resolve().parent.parent
DIR_PRED = RACINE / "data" / "predictions"
DIR_CACHE = RACINE / "data" / "cache"
SORTIE = RACINE / "docs" / "data.json"

# États NHL d'un match. Avant la mise au jeu : FUT (futur) ou PRE (pré-match).
# POURQUOI geler après la mise au jeu : une prédiction modifiée pendant le match
# serait de la triche dans le tracking. On ne touche plus à un match commencé.
ETATS_AVANT = {"FUT", "PRE"}
ETATS_FINIS = {"OFF", "FINAL"}

# Défauts pour une recrue sans aucun historique (minutes/match par position).
TOI_DEFAUT = {"F": 14.0, "D": 19.0}


# --- Utilitaires -----------------------------------------------------------------

def toi_min(txt):
    """'17:42' -> 17.7 minutes."""
    if not txt:
        return 0.0
    m, s = txt.split(":")
    return int(m) + int(s) / 60.0


def pos_de(code):
    """C/L/R -> 'F', D -> 'D'. On traite ailiers et centres pareil pour les moyennes."""
    return "D" if code == "D" else "F"


def saison_precedente(saison):
    """20262027 -> 20252026."""
    debut = saison // 10000
    return (debut - 1) * 10000 + debut


def lire_json(chemin, defaut):
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return defaut


def ecrire_json(chemin, data):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --- 1. Grader -------------------------------------------------------------------

def grader():
    """Remplit `result` (tirs réels) et `played` pour les matchs terminés."""
    for f in sorted(DIR_PRED.glob("*.json")):
        data = lire_json(f, None)
        if not data:
            continue
        a_grader = {e["gameId"] for e in data["entries"] if e.get("result") is None and e.get("played") is not False}
        if not a_grader:
            continue
        modifie = False
        for gid in a_grader:
            box = api.boxscore(gid)
            if not box or box.get("gameState") not in ETATS_FINIS:
                continue
            sog = {}
            for cote in ("awayTeam", "homeTeam"):
                stats = box.get("playerByGameStats", {}).get(cote, {})
                for grp in ("forwards", "defense"):
                    for j in stats.get(grp, []):
                        sog[j["playerId"]] = j.get("sog", 0)
            for e in data["entries"]:
                if e["gameId"] != gid:
                    continue
                if e["pid"] in sog:
                    e["result"], e["played"] = sog[e["pid"]], True
                else:
                    # Absent de la feuille de match = scratché/blessé.
                    # La prédiction est annulée, pas comptée comme un échec.
                    e["result"], e["played"] = 0, False
                modifie = True
        if modifie:
            ecrire_json(f, data)


# --- 2. Construire le contexte ---------------------------------------------------

def resumer_saison_prec(log):
    """Garde seulement ce dont le modèle a besoin de la saison passée.
    POURQUOI un cache résumé : la saison passée ne change plus. On la télécharge
    une fois par joueur et on garde ~100 octets au lieu de 20 Ko par joueur."""
    tot = {"shots": 0, "toi": 0.0, "gp": 0, "vs": {}}
    for g in (log or {}).get("gameLog", []):
        t = toi_min(g.get("toi"))
        tot["shots"] += g.get("shots", 0)
        tot["toi"] += t
        tot["gp"] += 1
        vs = tot["vs"].setdefault(g["opponentAbbrev"], [0, 0.0, 0])
        vs[0] += g.get("shots", 0)
        vs[1] += t
        vs[2] += 1
    tot["toi"] = round(tot["toi"], 2)
    for k in tot["vs"]:
        tot["vs"][k][1] = round(tot["vs"][k][1], 2)
    return tot


def tirs_accordes(team, saison=None):
    """Somme des tirs reçus par les gardiens de l'équipe."""
    cs = api.club_stats(team, saison)
    if not cs:
        return None
    return sum(g.get("shotsAgainst", 0) for g in cs.get("goalies", []))


def construire(jour, matchs_ouverts, saison):
    prev_saison = saison_precedente(saison)

    # Équipes + alignements
    equipes = {}
    for m in matchs_ouverts:
        equipes[m["away"]] = {"opp": m["home"], "home": False, "gameId": m["id"]}
        equipes[m["home"]] = {"opp": m["away"], "home": True, "gameId": m["id"]}

    joueurs = []
    for team in equipes:
        r = api.roster(team) or {}
        for grp in ("forwards", "defensemen"):
            for j in r.get(grp, []):
                joueurs.append({
                    "pid": j["id"],
                    "name": f'{j["firstName"]["default"]} {j["lastName"]["default"]}',
                    "team": team,
                    "pos": pos_de(j.get("positionCode")),
                })

    # Game logs de la saison en cours (en parallèle)
    chemins = {j["pid"]: api.game_log_path(j["pid"], saison) for j in joueurs}
    logs = api.fetch_many(list(chemins.values()))

    # Saison passée : depuis le cache, on télécharge seulement les manquants
    f_prev = DIR_CACHE / f"joueurs_{prev_saison}.json"
    cache_prev = lire_json(f_prev, {})
    manquants = [j["pid"] for j in joueurs if str(j["pid"]) not in cache_prev]
    if manquants:
        res = api.fetch_many([api.game_log_path(p, prev_saison) for p in manquants])
        for p in manquants:
            cache_prev[str(p)] = resumer_saison_prec(res[api.game_log_path(p, prev_saison)])
        ecrire_json(f_prev, cache_prev)

    # Tirs accordés : les 32 équipes, pour avoir une vraie moyenne de ligue
    st = api.standings_now() or {"standings": []}
    gp_equipe = {s["teamAbbrev"]["default"]: s["gamesPlayed"] for s in st["standings"]}
    f_eq = DIR_CACHE / f"equipes_{prev_saison}.json"
    cache_eq = lire_json(f_eq, {})
    sa = {}
    for team, gp in gp_equipe.items():
        if team not in cache_eq:
            tot_prev = tirs_accordes(team, prev_saison)
            # 82 matchs en saison régulière. Une équipe qui n'existait pas
            # la saison passée prend la moyenne (gérée plus bas avec None).
            cache_eq[team] = tot_prev / 82.0 if tot_prev else None
        tot_cur = tirs_accordes(team)
        sa[team] = {"gp": gp, "sa_cur": (tot_cur / gp) if gp and tot_cur else 0.0,
                    "sa_prev": cache_eq[team]}
    ecrire_json(f_eq, cache_eq)

    prev_vals = [v["sa_prev"] for v in sa.values() if v["sa_prev"]]
    sa_prev_moy = sum(prev_vals) / len(prev_vals) if prev_vals else 28.0
    for v in sa.values():
        if not v["sa_prev"]:
            v["sa_prev"] = sa_prev_moy
    # Moyenne de ligue sur le même mélange que celui appliqué à l'adversaire
    tot_gp = sum(v["gp"] for v in sa.values())
    cur_moy = sum(v["sa_cur"] * v["gp"] for v in sa.values()) / tot_gp if tot_gp else sa_prev_moy
    poids = tot_gp / len(sa) if sa else 0
    sa_avg = (poids * cur_moy + 15 * sa_prev_moy) / (poids + 15) if sa else 28.0

    # Parse des logs courants + dates de match de chaque équipe
    dates_equipe = {}
    for j in joueurs:
        log = logs.get(chemins[j["pid"]]) or {}
        games = []
        for g in log.get("gameLog", []):
            if g["gameDate"] >= jour:   # rien du futur, jamais
                continue
            games.append({"date": g["gameDate"], "shots": g.get("shots", 0),
                          "toi": toi_min(g.get("toi")), "opp": g["opponentAbbrev"]})
            dates_equipe.setdefault(g["teamAbbrev"], set()).add(g["gameDate"])
        games.sort(key=lambda g: g["date"])
        j["cur"] = games
        j["prev"] = cache_prev.get(str(j["pid"]), {"shots": 0, "toi": 0.0, "gp": 0, "vs": {}})

    # Moyennes de ligue par position (taux/min et TOI/match), sur le pool du soir
    league = {"sa_avg": sa_avg, "pos_rate": {}, "pos_toi": {}}
    for pos in ("F", "D"):
        s = t = n = 0.0
        for j in joueurs:
            if j["pos"] != pos:
                continue
            s += sum(g["shots"] for g in j["cur"]) + j["prev"]["shots"]
            t += sum(g["toi"] for g in j["cur"]) + j["prev"]["toi"]
            n += len(j["cur"]) + j["prev"]["gp"]
        league["pos_rate"][pos] = s / t if t else (0.12 if pos == "F" else 0.08)
        league["pos_toi"][pos] = t / n if n else TOI_DEFAUT[pos]

    hier = (date.fromisoformat(jour) - timedelta(days=1)).isoformat()
    ctxs = []
    for j in joueurs:
        team = j["team"]
        info = equipes[team]
        dates = sorted(dates_equipe.get(team, set()), reverse=True)
        derniers2 = set(dates[:2])

        # QUI JOUE CE SOIR? L'API ne donne pas l'alignement avant le match.
        # Heuristique : il doit avoir joué un des 2 derniers matchs de son équipe.
        # Ça élimine les blessés et les scratchés réguliers. Pas parfait (un gars
        # qui revient de blessure est exclu), d'où le grading qui annule les absents.
        if dates:
            if not j["cur"] or j["cur"][-1]["date"] not in derniers2:
                continue
            incertain = j["cur"][-1]["date"] != dates[0]
        else:
            incertain = False  # tout début de saison : tout le monde est éligible

        vs = {"shots": 0, "toi": 0.0, "gp": 0}
        for g in j["cur"]:
            if g["opp"] == info["opp"]:
                vs["shots"] += g["shots"]; vs["toi"] += g["toi"]; vs["gp"] += 1
        pv = j["prev"]["vs"].get(info["opp"])
        if pv:
            vs["shots"] += pv[0]; vs["toi"] += pv[1]; vs["gp"] += pv[2]

        ctx = {
            **j,
            "opp": dict(sa.get(info["opp"], {"gp": 0, "sa_cur": sa_avg, "sa_prev": sa_avg}),
                        abbrev=info["opp"]),
            "home": info["home"],
            "b2b": bool(dates) and dates[0] == hier,
            "vs": vs,
            "league": league,
            "gameId": info["gameId"],
            "incertain": incertain,
        }
        ctxs.append(ctx)
    return ctxs


# --- 3. Prédire et publier -------------------------------------------------------

def predire(ctxs):
    entries = []
    for ctx in ctxs:
        preds = {}
        for v, spec in VERSIONS.items():
            r = spec["fn"](dict(ctx))  # copie : une version ne pollue pas l'autre
            preds[v] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k != "detail"}
            if "detail" in r:
                preds[v]["detail"] = {k: round(x, 3) for k, x in r["detail"].items()}
        cur = ctx["cur"]
        l10 = cur[-10:]
        entries.append({
            "pid": ctx["pid"], "name": ctx["name"], "team": ctx["team"],
            "opp": ctx["opp"]["abbrev"], "home": ctx["home"], "pos": ctx["pos"],
            "gameId": ctx["gameId"], "b2b": ctx["b2b"], "incertain": ctx["incertain"],
            # Repère visuel : combien de fois 2+ tirs sur les 10 derniers matchs
            "l10_hits": sum(1 for g in l10 if g["shots"] >= 2), "l10_n": len(l10),
            "l10_shots": [g["shots"] for g in l10],
            "vs_gp": ctx["vs"]["gp"], "vs_shots": ctx["vs"]["shots"],
            "preds": preds,
            "result": None, "played": None,
        })
    return entries


def jour_nhl():
    """La NHL date ses matchs à l'heure de l'Est (même fuseau que La Tuque)."""
    return datetime.now(ZoneInfo("America/Toronto")).date().isoformat()


def main():
    jour = sys.argv[1] if len(sys.argv) > 1 else jour_nhl()
    DIR_PRED.mkdir(parents=True, exist_ok=True)

    grader()

    sched = api.schedule(jour) or {}
    semaine = sched.get("gameWeek", [])
    matchs_jour = next((d["games"] for d in semaine if d["date"] == jour), [])
    matchs = [{
        "id": g["id"], "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
        "start": g["startTimeUTC"], "state": g.get("gameState", "FUT"), "season": g.get("season"),
    } for g in matchs_jour if g.get("gameType") in (2, None)]

    f_jour = DIR_PRED / f"{jour}.json"
    existant = lire_json(f_jour, {"date": jour, "entries": []})
    ouverts = [m for m in matchs if m["state"] in ETATS_AVANT]

    if ouverts:
        saison = ouverts[0]["season"]
        nouvelles = predire(construire(jour, ouverts, saison))
        ids_ouverts = {m["id"] for m in ouverts}
        # On garde intactes les prédictions des matchs déjà commencés (gelées).
        gardees = [e for e in existant["entries"] if e["gameId"] not in ids_ouverts]
        existant["entries"] = gardees + nouvelles
        existant["versions"] = list(VERSIONS)
        existant["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
        ecrire_json(f_jour, existant)

    historique = [lire_json(f, None) for f in sorted(DIR_PRED.glob("*.json"))]
    historique = [h for h in historique if h]

    ecrire_json(SORTIE, {
        "genere": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "date": jour,
        "versions": [{"id": v, "desc": s["desc"]} for v, s in VERSIONS.items()],
        "principale": VERSION_PRINCIPALE,
        "matchs": matchs,
        "joueurs": existant["entries"],
        "stats": resumer(historique, list(VERSIONS)),
    })
    print(f"{jour}: {len(matchs)} matchs, {len(existant['entries'])} joueurs, "
          f"{sum(1 for h in historique for e in h['entries'] if e.get('played'))} prédictions gradées au total")


if __name__ == "__main__":
    main()
