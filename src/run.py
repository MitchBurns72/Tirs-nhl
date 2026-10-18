"""
Pipeline quotidien. Roule dans la GitHub Action (3 fois par jour) :

  1. GRADER   : les prédictions des soirs passés dont les matchs sont finis
                reçoivent leurs résultats (tirs, buts, passes, score final).
  2. PRÉDIRE  : pour chaque match de ce soir pas encore commencé, toutes les
                versions calculent : victoire, total de buts, et pour chaque
                joueur P(2+ tirs), P(1+ but), P(1+ passe).
  3. COTES    : (calcul de 17 h 45 seulement) cotes US via The Odds API.
  4. PUBLIER  : docs/data.json est réécrit; la page le lit.

Lancement : python src/run.py            (date du jour, heure de l'Est)
            python src/run.py 2026-10-18 (forcer une date, pratique pour tester)
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import auto
import cotes
import nhl_api as api
from modeles import (DISPERSION, VERSIONS, ajouter_seuils, charger_auto, features_joueur,
                     features_match, modele_joueur_param, modele_match_param)
from score import resumer

RACINE = Path(__file__).resolve().parent.parent
DIR_PRED = RACINE / "data" / "predictions"
DIR_CACHE = RACINE / "data" / "cache"
SORTIE = RACINE / "docs" / "data.json"

# Avant la mise au jeu : FUT ou PRE. POURQUOI geler après : une prédiction
# modifiée pendant le match serait de la triche dans le tracking.
ETATS_AVANT = {"FUT", "PRE"}
ETATS_FINIS = {"OFF", "FINAL"}
TOI_DEFAUT = {"F": 14.0, "D": 19.0}

# Les cotes coûtent des crédits : seulement quand le workflow le demande
# (calcul de 17 h 45 ou lancement manuel). Voir cotes.py pour le budget.
AVEC_COTES = os.environ.get("ODDS", "").lower() == "true"
MARCHES_PROPS = [m for m in os.environ.get("ODDS_PROPS", "player_shots_on_goal_alternate").split(",") if m]


# --- Utilitaires -----------------------------------------------------------------

def toi_min(txt):
    """'17:42' -> 17.7 minutes."""
    if not txt:
        return 0.0
    m, s = txt.split(":")
    return int(m) + int(s) / 60.0


def pos_de(code):
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


def arrondir(x):
    """Arrondit récursivement pour garder les fichiers JSON lisibles et petits."""
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: arrondir(v) for k, v in x.items()}
    if isinstance(x, list):
        return [arrondir(v) for v in x]
    return x


# --- 1. Grader -------------------------------------------------------------------

def grader():
    for f in sorted(DIR_PRED.glob("*.json")):
        data = lire_json(f, None)
        if not data:
            continue
        data.setdefault("games", [])
        # Une entrée de l'ancien format a `result` = un nombre (tirs seulement).
        # On ne regrade que ce qui n'a pas encore de résultat.
        a_grader = {e["gameId"] for e in data["entries"] if e.get("result") is None and e.get("played") is not False}
        a_grader |= {g["gameId"] for g in data["games"] if g.get("result") is None}
        modifie = False
        for gid in a_grader:
            box = api.boxscore(gid)
            if not box or box.get("gameState") not in ETATS_FINIS:
                continue
            stats, buts_eq = {}, {}
            for cote in ("awayTeam", "homeTeam"):
                bloc = box.get("playerByGameStats", {}).get(cote, {})
                buts_eq[cote] = 0
                for grp in ("forwards", "defense"):
                    for j in bloc.get(grp, []):
                        stats[j["playerId"]] = {"tirs": j.get("sog", 0), "buts": j.get("goals", 0),
                                                "passes": j.get("assists", 0)}
                        buts_eq[cote] += j.get("goals", 0)
            for e in data["entries"]:
                if e["gameId"] != gid or e.get("result") is not None:
                    continue
                if e["pid"] in stats:
                    e["result"], e["played"] = stats[e["pid"]], True
                else:
                    # Absent de la feuille de match = scratché/blessé : annulé, pas un échec.
                    e["result"], e["played"] = {}, False
                modifie = True
            for g in data["games"]:
                if g["gameId"] == gid and g.get("result") is None:
                    # Score officiel (inclut le but de fusillade); sinon somme des buts.
                    sh = box.get("homeTeam", {}).get("score", buts_eq["homeTeam"])
                    sa = box.get("awayTeam", {}).get("score", buts_eq["awayTeam"])
                    g["result"] = {"home": sh, "away": sa}
                    modifie = True
        if modifie:
            ecrire_json(f, data)


# --- 2. Contexte des équipes (modèle de match) --------------------------------------

def resume_calendrier(cal, team, avant):
    """Liste des matchs de saison régulière terminés avant `avant`, du point de vue de `team`."""
    jeux = []
    for g in (cal or {}).get("games", []):
        if g.get("gameType") != 2 or g.get("gameState") not in ETATS_FINIS or g["gameDate"] >= avant:
            continue
        home = g["homeTeam"]["abbrev"] == team
        moi, eux = (g["homeTeam"], g["awayTeam"]) if home else (g["awayTeam"], g["homeTeam"])
        jeux.append({"date": g["gameDate"], "gf": moi.get("score", 0), "ga": eux.get("score", 0),
                     "opp": eux["abbrev"], "home": home})
    jeux.sort(key=lambda j: j["date"])
    return jeux


def contexte_equipes(jour, equipes, saison):
    """Pour chaque équipe du soir : matchs de la saison (scores), résumé de la
    saison passée (cache), et duels directs."""
    prev_saison = saison_precedente(saison)
    f_cache = DIR_CACHE / f"calendriers_{prev_saison}.json"
    cache = lire_json(f_cache, {})
    out = {}
    for t in equipes:
        cur = resume_calendrier(api.club_schedule(t), t, jour)
        if t not in cache:
            # La saison passée ne change plus : on la résume une fois et on la garde.
            jeux = resume_calendrier(api.club_schedule(t, prev_saison), t, "9999")
            vs = {}
            for j in jeux:
                d = vs.setdefault(j["opp"], [0, 0, 0])
                d[0] += j["gf"]; d[1] += j["ga"]; d[2] += 1
            cache[t] = {"gp": len(jeux), "gf": sum(j["gf"] for j in jeux),
                        "ga": sum(j["ga"] for j in jeux), "vs": vs}
        out[t] = {"cur": cur, "prev": cache[t]}
    ecrire_json(f_cache, cache)
    return out, cache


# --- 3. Contexte des joueurs -----------------------------------------------------

def resumer_saison_prec(log):
    """Garde seulement ce dont le modèle a besoin de la saison passée (~150 octets/joueur)."""
    tot = {"shots": 0, "goals": 0, "assists": 0, "toi": 0.0, "gp": 0, "vs": {}}
    for g in (log or {}).get("gameLog", []):
        t = toi_min(g.get("toi"))
        tot["shots"] += g.get("shots", 0)
        tot["goals"] += g.get("goals", 0)
        tot["assists"] += g.get("assists", 0)
        tot["toi"] += t
        tot["gp"] += 1
        vs = tot["vs"].setdefault(g["opponentAbbrev"], [0, 0.0, 0])
        vs[0] += g.get("shots", 0); vs[1] += t; vs[2] += 1
    tot["toi"] = round(tot["toi"], 2)
    for k in tot["vs"]:
        tot["vs"][k][1] = round(tot["vs"][k][1], 2)
    return tot


def stats_gardiens(team, saison=None):
    """(tirs accordés, buts accordés) des gardiens de l'équipe."""
    cs = api.club_stats(team, saison)
    if not cs:
        return None, None
    g = cs.get("goalies", [])
    return sum(x.get("shotsAgainst", 0) for x in g), sum(x.get("goalsAgainst", 0) for x in g)


def resume_box(box):
    """Version compacte d'une feuille de match pour le cache de la saison.
    Fallbacks : si le score ou les tirs d'équipe manquent, on additionne les joueurs."""
    pb = box.get("playerByGameStats", {})
    def somme(cote, champ):
        return sum(j.get(champ, 0) for grp in ("forwards", "defense") for j in pb.get(cote, {}).get(grp, []))
    h, a = box.get("homeTeam", {}), box.get("awayTeam", {})
    return {"date": box.get("gameDate", ""), "home": h.get("abbrev"), "away": a.get("abbrev"),
            "hg": h.get("score", somme("homeTeam", "goals")), "ag": a.get("score", somme("awayTeam", "goals")),
            "hsog": h.get("sog", somme("homeTeam", "sog")), "asog": a.get("sog", somme("awayTeam", "sog")),
            "fin": box.get("gameOutcome", {}).get("lastPeriodType", "REG")}


def cache_feuilles(saison, equipes):
    """Toutes les feuilles de match terminées de la saison, en cache permanent.

    POURQUOI : pour rejouer un soir passé SANS TRICHER, il faut les tirs et buts
    accordés par chaque équipe AVANT ce soir-là. Les stats d'équipe « à date »
    de l'API incluent tout jusqu'à aujourd'hui (fuite du futur). Avec les feuilles
    de match datées, on filtre par date et on reconstruit l'état exact de n'importe
    quel soir. Une feuille finie ne change plus : on la télécharge une seule fois.
    """
    f = DIR_CACHE / f"feuilles_{saison}.json"
    cache = lire_json(f, {})
    ids = set()
    for t in equipes:
        for g in (api.club_schedule(t) or {}).get("games", []):
            if g.get("gameType") == 2 and g.get("gameState") in ETATS_FINIS:
                ids.add(g["id"])
    manquants = [i for i in ids if str(i) not in cache]
    res = api.fetch_many([f"/gamecenter/{i}/boxscore" for i in manquants])
    for i in manquants:
        b = res.get(f"/gamecenter/{i}/boxscore")
        if b and b.get("gameState") in ETATS_FINIS:
            cache[str(i)] = resume_box(b)
    ecrire_json(f, cache)
    return cache


def contexte_ligue(jour, saison):
    """Tirs accordés et buts accordés par tir pour les 32 équipes + moyenne de buts,
    tels qu'ils étaient AVANT `jour` (voir cache_feuilles)."""
    prev_saison = saison_precedente(saison)
    st = api.standings_now() or {"standings": []}
    noms = {s["teamAbbrev"]["default"]: s.get("teamCommonName", {}).get("default", "") for s in st["standings"]}
    feuilles = cache_feuilles(saison, list(noms))

    tot = {t: {"gp": 0, "sa": 0, "ga": 0, "gf": 0} for t in noms}
    for fb in feuilles.values():
        if fb["date"] >= jour or fb["home"] not in tot or fb["away"] not in tot:
            continue
        # But de fusillade : compte dans le score officiel mais pas comme but
        # accordé par le gardien. On le retire pour le % d'arrêt.
        so_h = 1 if fb["fin"] == "SO" and fb["hg"] > fb["ag"] else 0
        so_a = 1 if fb["fin"] == "SO" and fb["ag"] > fb["hg"] else 0
        for moi, eux, gf, ga, sa in ((fb["home"], fb["away"], fb["hg"], fb["ag"] - so_a, fb["asog"]),
                                    (fb["away"], fb["home"], fb["ag"], fb["hg"] - so_h, fb["hsog"])):
            t = tot[moi]
            t["gp"] += 1; t["sa"] += sa; t["ga"] += ga; t["gf"] += gf
    gp_tot = sum(t["gp"] for t in tot.values())
    gf_tot = sum(t["gf"] for t in tot.values())

    f_eq = DIR_CACHE / f"gardiens_{prev_saison}.json"
    cache = lire_json(f_eq, {})
    eq = {}
    for team, t in tot.items():
        if team not in cache:
            sa_p, ga_p = stats_gardiens(team, prev_saison)
            # 82 matchs en saison régulière. Équipe sans saison passée -> moyenne plus bas.
            cache[team] = {"sa_pg": sa_p / 82.0 if sa_p else None,
                           "gaps": ga_p / sa_p if sa_p else None}
        eq[team] = {"gp": t["gp"],
                    "sa_cur": t["sa"] / t["gp"] if t["gp"] else 0.0, "sa_prev": cache[team]["sa_pg"],
                    "gaps_cur": t["ga"] / t["sa"] if t["sa"] else 0.0, "gaps_prev": cache[team]["gaps"]}
    ecrire_json(f_eq, cache)

    def moyenne_prev(cle, defaut):
        vals = [v[cle] for v in eq.values() if v[cle]]
        return sum(vals) / len(vals) if vals else defaut
    sa_prev_moy, gaps_prev_moy = moyenne_prev("sa_prev", 28.0), moyenne_prev("gaps_prev", 0.10)
    for v in eq.values():
        v["sa_prev"] = v["sa_prev"] or sa_prev_moy
        v["gaps_prev"] = v["gaps_prev"] or gaps_prev_moy
        if not v["gp"]:
            v["sa_cur"], v["gaps_cur"] = v["sa_prev"], v["gaps_prev"]

    # Moyennes de ligue sur le même mélange (15 matchs de saison passée) que l'adversaire.
    poids = gp_tot / len(eq) if eq else 0
    cur_sa = sum(v["sa_cur"] * v["gp"] for v in eq.values()) / gp_tot if gp_tot else sa_prev_moy
    cur_gaps = sum(v["gaps_cur"] * v["gp"] for v in eq.values()) / gp_tot if gp_tot else gaps_prev_moy
    melange = lambda c, p: (poids * c + 15 * p) / (poids + 15)
    # `mu` (buts par équipe par match) est finalisé dans predire(), une fois la
    # moyenne de la saison passée connue. Ici on garde juste la partie courante.
    return eq, noms, {"sa_avg": melange(cur_sa, sa_prev_moy), "gaps_avg": melange(cur_gaps, gaps_prev_moy),
                      "mu_cur": gf_tot / gp_tot if gp_tot else None, "poids": poids}


def construire_joueurs(jour, equipes, saison, ligue_eq, ligue, ctx_eq):
    prev_saison = saison_precedente(saison)
    joueurs = []
    for team in equipes:
        r = api.roster(team) or {}
        for grp in ("forwards", "defensemen"):
            for j in r.get(grp, []):
                joueurs.append({"pid": j["id"], "team": team, "pos": pos_de(j.get("positionCode")),
                                "name": f'{j["firstName"]["default"]} {j["lastName"]["default"]}'})

    chemins = {j["pid"]: api.game_log_path(j["pid"], saison) for j in joueurs}
    logs = api.fetch_many(list(chemins.values()))

    # Nouveau nom de cache : l'ancien n'avait pas les buts/passes.
    f_prev = DIR_CACHE / f"joueurs2_{prev_saison}.json"
    cache_prev = lire_json(f_prev, {})
    manquants = [j["pid"] for j in joueurs if str(j["pid"]) not in cache_prev]
    if manquants:
        res = api.fetch_many([api.game_log_path(p, prev_saison) for p in manquants])
        for p in manquants:
            cache_prev[str(p)] = resumer_saison_prec(res[api.game_log_path(p, prev_saison)])
        ecrire_json(f_prev, cache_prev)

    for j in joueurs:
        log = logs.get(chemins[j["pid"]]) or {}
        games = [{"date": g["gameDate"], "shots": g.get("shots", 0), "goals": g.get("goals", 0),
                  "assists": g.get("assists", 0), "toi": toi_min(g.get("toi")), "opp": g["opponentAbbrev"]}
                 for g in log.get("gameLog", []) if g["gameDate"] < jour]   # rien du futur, jamais
        games.sort(key=lambda g: g["date"])
        j["cur"] = games
        j["prev"] = cache_prev.get(str(j["pid"]), {"shots": 0, "goals": 0, "assists": 0, "toi": 0.0, "gp": 0, "vs": {}})

    # Moyennes de ligue par position, sur le pool du soir (saison + passée)
    ligue = dict(ligue, pos_rate={}, pos_toi={}, pos_sh={}, pos_pa_rate={}, pos_par_match={})
    for pos in ("F", "D"):
        s = b = a = t = n = 0.0
        for j in joueurs:
            if j["pos"] != pos:
                continue
            p = j["prev"]
            s += sum(g["shots"] for g in j["cur"]) + p["shots"]
            b += sum(g["goals"] for g in j["cur"]) + p.get("goals", 0)
            a += sum(g["assists"] for g in j["cur"]) + p.get("assists", 0)
            t += sum(g["toi"] for g in j["cur"]) + p["toi"]
            n += len(j["cur"]) + p["gp"]
        ligue["pos_rate"][pos] = s / t if t else (0.12 if pos == "F" else 0.08)
        ligue["pos_toi"][pos] = t / n if n else TOI_DEFAUT[pos]
        ligue["pos_sh"][pos] = b / s if s else (0.10 if pos == "F" else 0.045)
        ligue["pos_pa_rate"][pos] = a / t if t else (0.025 if pos == "F" else 0.02)
        ligue["pos_par_match"][pos] = {"goals": b / n if n else 0.15, "assists": a / n if n else 0.2}

    hier = (date.fromisoformat(jour) - timedelta(days=1)).isoformat()
    ctxs = []
    for j in joueurs:
        info = equipes[j["team"]]
        dates = sorted({g["date"] for g in ctx_eq[j["team"]]["cur"]}, reverse=True)
        derniers2 = set(dates[:2])
        # QUI JOUE CE SOIR? L'API ne donne pas l'alignement avant le match.
        # Heuristique : il doit avoir joué un des 2 derniers matchs de son équipe.
        # Les scratchés qui passent quand même sont annulés au grading.
        if dates:
            if not j["cur"] or j["cur"][-1]["date"] not in derniers2:
                continue
            incertain = j["cur"][-1]["date"] != dates[0]
        else:
            incertain = False

        vs = {"shots": 0, "toi": 0.0, "gp": 0}
        for g in j["cur"]:
            if g["opp"] == info["opp"]:
                vs["shots"] += g["shots"]; vs["toi"] += g["toi"]; vs["gp"] += 1
        pv = j["prev"]["vs"].get(info["opp"])
        if pv:
            vs["shots"] += pv[0]; vs["toi"] += pv[1]; vs["gp"] += pv[2]

        defaut = {"gp": 0, "sa_cur": ligue["sa_avg"], "sa_prev": ligue["sa_avg"],
                  "gaps_cur": ligue["gaps_avg"], "gaps_prev": ligue["gaps_avg"]}
        ctxs.append({**j,
                     "opp": dict(ligue_eq.get(info["opp"], defaut), abbrev=info["opp"]),
                     "home": info["home"], "b2b": bool(dates) and dates[0] == hier,
                     "vs": vs, "league": ligue, "gameId": info["gameId"], "incertain": incertain})
    return ctxs


# --- 4. Prédire ------------------------------------------------------------------

def predire(jour, matchs_ouverts, saison):
    equipes = {}
    for m in matchs_ouverts:
        equipes[m["away"]] = {"opp": m["home"], "home": False, "gameId": m["id"]}
        equipes[m["home"]] = {"opp": m["away"], "home": True, "gameId": m["id"]}

    ligue_eq, noms, ligue = contexte_ligue(jour, saison)
    ctx_eq, cache_prev = contexte_equipes(jour, equipes, saison)

    # Buts par équipe par match. POURQUOI pas une constante : le niveau d'offensive
    # de la ligue bouge d'une saison à l'autre (règles, arbitrage). Si `mu` est
    # décalé, toutes les attaques/défenses relatives le sont aussi, et le total
    # projeté + les passes sont faussés. On mélange la saison passée (moyenne des
    # équipes en cache) et la saison courante, comme pour les tirs.
    gp_p = sum(c["gp"] for c in cache_prev.values())
    mu_prev = sum(c["gf"] for c in cache_prev.values()) / gp_p if gp_p else 3.05
    mu_cur = ligue["mu_cur"] if ligue["mu_cur"] is not None else mu_prev
    ligue["mu"] = (ligue["poids"] * mu_cur + 15 * mu_prev) / (ligue["poids"] + 15)
    hier = (date.fromisoformat(jour) - timedelta(days=1)).isoformat()

    # Une version auto ne prédit jamais un soir antérieur à sa création : ces soirs
    # ont servi à l'entraîner, ses prédictions y seraient de la triche (in-sample).
    actives = {v: spec for v, spec in VERSIONS.items() if not spec.get("auto") or spec["cree"] <= jour}

    # Matchs d'abord : les passes des joueurs dépendent des buts attendus de l'équipe.
    games, jeu_par_equipe = [], {v: {} for v in actives}
    for m in matchs_ouverts:
        cm = {"mu": ligue["mu"]}
        for cote in ("home", "away"):
            t = m[cote]
            cur = ctx_eq[t]["cur"]
            opp = m["away"] if cote == "home" else m["home"]
            h2h = {"gf": sum(g["gf"] for g in cur if g["opp"] == opp),
                   "gp": sum(1 for g in cur if g["opp"] == opp)}
            pv = ctx_eq[t]["prev"]["vs"].get(opp)
            if pv:
                h2h["gf"] += pv[0]; h2h["gp"] += pv[2]
            cm[cote] = {"abbrev": t, "cur": cur, "prev": ctx_eq[t]["prev"], "h2h": h2h,
                        "cur_gp": len(cur), "cur_gf": sum(g["gf"] for g in cur), "cur_ga": sum(g["ga"] for g in cur),
                        "b2b": bool(cur) and cur[-1]["date"] == hier}
        preds = {}
        fm = features_match(cm)
        for v, spec in actives.items():
            # Version auto : même modèle que v1, paramètres trouvés par le tuner
            r = modele_match_param(fm, spec["params"]) if spec.get("auto") else spec["match"](cm)
            preds[v] = r
            jeu_par_equipe[v][m["home"]] = {"lam_pour": r["lam_home"], "lam_contre": r["lam_away"]}
            jeu_par_equipe[v][m["away"]] = {"lam_pour": r["lam_away"], "lam_contre": r["lam_home"]}
        l10 = lambda c: cm[c]["cur"][-10:]
        games.append({
            "gameId": m["id"], "home": m["home"], "away": m["away"], "start": m["start"],
            "b2b_home": cm["home"]["b2b"], "b2b_away": cm["away"]["b2b"],
            "forme": {c: {"gf": sum(g["gf"] for g in l10(c)), "ga": sum(g["ga"] for g in l10(c)),
                          "n": len(l10(c)),
                          "v": sum(1 for g in l10(c) if g["gf"] > g["ga"])} for c in ("home", "away")},
            "h2h": {"gp": cm["home"]["h2h"]["gp"], "gf_home": cm["home"]["h2h"]["gf"],
                    "gf_away": cm["away"]["h2h"]["gf"]},
            "preds": arrondir(preds), "result": None,
            "f": arrondir(fm),   # journal de features : sert au tuner (auto.py)
        })

    entries = []
    for ctx in construire_joueurs(jour, equipes, saison, ligue_eq, ligue, ctx_eq):
        # Features calculées avec les buts attendus de v1 : c'est la référence
        # commune sur laquelle le tuner rejoue toutes les variantes.
        f = features_joueur(ctx, jeu_par_equipe.get("v1", {}).get(ctx["team"]))
        preds = {v: (modele_joueur_param(f, spec["params"]) if spec.get("auto")
                     else spec["joueur"](dict(ctx), jeu_par_equipe[v].get(ctx["team"])))
                 for v, spec in actives.items()}
        for v, spec in actives.items():
            ajouter_seuils(preds[v], spec.get("nb_k"))   # tirs 1+, 3+, 4+
        l10 = ctx["cur"][-10:]
        entries.append({
            "pid": ctx["pid"], "name": ctx["name"], "team": ctx["team"],
            "opp": ctx["opp"]["abbrev"], "home": ctx["home"], "pos": ctx["pos"],
            "gameId": ctx["gameId"], "b2b": ctx["b2b"], "incertain": ctx["incertain"],
            # Repères visuels : les 10 derniers matchs
            "l10_shots": [g["shots"] for g in l10],
            "l10_goals": [g["goals"] for g in l10],
            "l10_assists": [g["assists"] for g in l10],
            "vs_gp": ctx["vs"]["gp"], "vs_shots": ctx["vs"]["shots"],
            "preds": arrondir(preds), "result": None, "played": None,
            "f": arrondir(f),
        })
    return games, entries, noms


# --- 5. Main ---------------------------------------------------------------------

MAX_RETRO = 40   # soirs rejoués max par exécution (garde l'Action sous ~10 min)


def retroactif(jour, saison):
    """Rejoue les soirs passés de la saison qui n'ont pas de fichier de prédictions.

    POURQUOI c'est légitime (pas de triche) : chaque soir rejoué utilise seulement
    ce qui était connu AVANT ce soir-là (game logs < date, calendriers < date,
    feuilles de match < date). C'est un vrai test hors-échantillon, ce qui nous
    donne des centaines de prédictions gradées dès le départ pour les notes.
    Seule exception assumée : l'alignement utilisé est celui d'aujourd'hui. Un gars
    échangé ou renvoyé dans les mineures depuis n'apparaît pas dans les vieux soirs.
    Les fichiers rejoués sont marqués "retro": true.
    """
    st = api.standings_now() or {"standings": []}
    par_date = {}
    for s in st["standings"]:
        t = s["teamAbbrev"]["default"]
        for g in (api.club_schedule(t) or {}).get("games", []):
            if g.get("gameType") != 2 or g.get("gameState") not in ETATS_FINIS or g["gameDate"] >= jour:
                continue
            par_date.setdefault(g["gameDate"], {})[g["id"]] = {
                "id": g["id"], "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                "start": g.get("startTimeUTC", g["gameDate"] + "T23:00:00Z"), "state": "FUT", "season": saison}
    manquants = [d for d in sorted(par_date) if not (DIR_PRED / f"{d}.json").exists()][:MAX_RETRO]
    for d in manquants:
        matchs = list(par_date[d].values())
        games, entries, _ = predire(d, matchs, saison)
        ecrire_json(DIR_PRED / f"{d}.json", {
            "date": d, "retro": True, "versions": list(VERSIONS),
            "maj": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "entries": entries, "games": games})
        print(f"  rétroactif {d}: {len(matchs)} matchs, {len(entries)} joueurs")
    return len(manquants)


def jour_nhl():
    """La NHL date ses matchs à l'heure de l'Est (même fuseau que La Tuque)."""
    return datetime.now(ZoneInfo("America/Toronto")).date().isoformat()


ETAT_AUTO = RACINE / "data" / "versions_auto.json"
MAX_COMPLETER = 10   # fichiers complétés par exécution


def completer_features(saison):
    """Ajoute le journal de features aux vieilles prédictions qui n'en ont pas
    (celles d'avant l'auto-upgrade), pour que le tuner ait du data dès le départ.

    POURQUOI c'est honnête : on recalcule seulement le CONTEXTE d'avant-match
    (filtré < date, comme le rétroactif). Les prédictions déjà faites ne sont
    jamais modifiées, on y colle juste les features."""
    n = 0
    for fic in sorted(DIR_PRED.glob("*.json")):
        if n >= MAX_COMPLETER:
            break
        data = lire_json(fic, None)
        if not data:
            continue
        manque_j = [e for e in data["entries"] if not e.get("f")]
        manque_g = [g for g in data.get("games", []) if not g.get("f")]
        if not manque_j and not manque_g:
            continue
        # Reconstituer les matchs du soir à partir des joueurs (domicile = équipe du joueur si home)
        matchs = {}
        for e in data["entries"]:
            h, a = (e["team"], e["opp"]) if e["home"] else (e["opp"], e["team"])
            matchs[e["gameId"]] = {"id": e["gameId"], "home": h, "away": a, "state": "OFF", "season": saison,
                                   "start": data["date"] + "T23:00:00Z"}
        for g in data.get("games", []):
            matchs[g["gameId"]] = {"id": g["gameId"], "home": g["home"], "away": g["away"], "state": "OFF",
                                   "season": saison, "start": g.get("start", data["date"] + "T23:00:00Z")}
        if not matchs:
            continue
        games, entries, _ = predire(data["date"], list(matchs.values()), saison)
        f_j = {(e["gameId"], e["pid"]): e["f"] for e in entries}
        f_g = {g["gameId"]: g["f"] for g in games}
        for e in manque_j:
            e["f"] = f_j.get((e["gameId"], e["pid"]))
        for g in manque_g:
            g["f"] = f_g.get(g["gameId"])
        ecrire_json(fic, data)
        n += 1
        print(f"  features ajoutées : {data['date']}")


def completer_seuils():
    """Ajoute tirs 1+/3+/4+ aux prédictions qui ne les ont pas (faites avant
    l'ajout de ces marchés). Calculé à partir des tirs attendus sauvés avant le
    match : aucune donnée d'après-match. Rapide, aucun appel réseau."""
    for fic in sorted(DIR_PRED.glob("*.json")):
        data = lire_json(fic, None)
        if not data:
            continue
        modifie = False
        for e in data["entries"]:
            for v, pv in e.get("preds", {}).items():
                if v in DISPERSION and ajouter_seuils(pv, DISPERSION[v]):
                    modifie = True
        if modifie:
            ecrire_json(fic, data)


def sans_features(liste):
    """La page n'a pas besoin des features (lourdes) : on les retire de data.json."""
    return [{k: v for k, v in x.items() if k != "f"} for x in liste]


def main():
    jour = sys.argv[1] if len(sys.argv) > 1 else jour_nhl()
    DIR_PRED.mkdir(parents=True, exist_ok=True)
    principale = charger_auto(ETAT_AUTO)   # ajoute les versions auto au registre
    completer_seuils()

    grader()

    sched = api.schedule(jour) or {}
    matchs_jour = next((d["games"] for d in sched.get("gameWeek", []) if d["date"] == jour), [])
    matchs = [{"id": g["id"], "away": g["awayTeam"]["abbrev"], "home": g["homeTeam"]["abbrev"],
               "start": g["startTimeUTC"], "state": g.get("gameState", "FUT"), "season": g.get("season")}
              for g in matchs_jour if g.get("gameType") in (2, None)]

    # Saison courante : celle des matchs du jour, sinon celle du calendrier d'une équipe
    saison = matchs[0]["season"] if matchs else (api.club_schedule("TOR") or {}).get("currentSeason")
    if saison and retroactif(jour, saison):
        grader()   # les soirs rejoués sont gradés tout de suite
    if saison:
        completer_features(saison)

    # Labo : patterns + tuning hebdo + promotion. Si une nouvelle version naît,
    # elle est chargée tout de suite et prédit déjà les matchs de ce soir.
    historique = [h for h in (lire_json(f, None) for f in sorted(DIR_PRED.glob("*.json"))) if h]
    labo, nouvelle = auto.cycle(historique, ETAT_AUTO, list(VERSIONS), jour)
    if nouvelle:
        charger_auto(ETAT_AUTO)
    principale = labo.get("principale", principale) if labo.get("principale") in VERSIONS else principale

    f_jour = DIR_PRED / f"{jour}.json"
    existant = lire_json(f_jour, {"date": jour, "entries": [], "games": []})
    existant.setdefault("games", [])
    ouverts = [m for m in matchs if m["state"] in ETATS_AVANT]
    ids_ouverts = {m["id"] for m in ouverts}
    cotes_m, cotes_j = {}, {}

    # Matchs de ce soir déjà commencés mais jamais prédits (ex. calcul lancé à
    # 21 h : les matchs de 13 h et 19 h sont déjà partis). On les prédit avec les
    # données d'AVANT ce soir (tout est filtré < jour), donc pas de triche : même
    # principe que le rétroactif des soirs passés. Marqués "retro": true.
    deja_j = {e["gameId"] for e in existant["entries"]}
    deja_g = {g["gameId"] for g in existant["games"]}
    commences_sans = [m for m in matchs if m["state"] not in ETATS_AVANT
                      and (m["id"] not in deja_j or m["id"] not in deja_g)]
    ids_retro = {m["id"] for m in commences_sans}
    a_predire = ouverts + commences_sans

    if a_predire:
        games, entries, noms = predire(jour, a_predire, a_predire[0]["season"])
        # Rétro : on ajoute seulement ce qui manque (joueurs et/ou match), jamais
        # on ne remplace une prédiction faite avant la mise au jeu.
        games_retro = [dict(g, retro=True) for g in games if g["gameId"] in ids_retro and g["gameId"] not in deja_g]
        entries_retro = [dict(e, retro=True) for e in entries if e["gameId"] in ids_retro and e["gameId"] not in deja_j]
        games = [g for g in games if g["gameId"] in ids_ouverts]
        entries = [e for e in entries if e["gameId"] in ids_ouverts]
        if AVEC_COTES and ouverts:
            cotes_m, cotes_j = cotes.recuperer(ouverts, noms, MARCHES_PROPS)
            for g in games:
                g["cotes"] = cotes_m.get(g["gameId"])
            for e in entries:
                e["cotes"] = cotes_j.get((e["gameId"], cotes.norm(e["name"])))
        else:
            # Pas d'appel de cotes à ce calcul : on garde celles du calcul précédent.
            anciennes_g = {g["gameId"]: g.get("cotes") for g in existant["games"]}
            anciennes_j = {(e["gameId"], e["pid"]): e.get("cotes") for e in existant["entries"]}
            for g in games:
                g["cotes"] = anciennes_g.get(g["gameId"])
            for e in entries:
                e["cotes"] = anciennes_j.get((e["gameId"], e["pid"]))
        # Les matchs déjà commencés gardent leurs prédictions gelées.
        existant["entries"] = [e for e in existant["entries"] if e["gameId"] not in ids_ouverts] + entries_retro + entries
        existant["games"] = [g for g in existant["games"] if g["gameId"] not in ids_ouverts] + games_retro + games
        existant["versions"] = list(VERSIONS)
        existant["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
        ecrire_json(f_jour, existant)
        if games_retro or entries_retro:
            print(f"  rétro ce soir : {len(games_retro)} matchs, {len(entries_retro)} joueurs")
            grader()   # ceux qui sont déjà finis sont gradés tout de suite
            existant = lire_json(f_jour, existant)

    historique = [h for h in (lire_json(f, None) for f in sorted(DIR_PRED.glob("*.json"))) if h]
    quota = lire_json(SORTIE, {}).get("cotes_quota")
    if AVEC_COTES:
        quota = {**cotes.etat, "quand": datetime.now(timezone.utc).isoformat(timespec="minutes")}

    ecrire_json(SORTIE, {
        "genere": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "date": jour,
        "versions": [{"id": v, "desc": s["desc"]} for v, s in VERSIONS.items()],
        "principale": principale,
        "matchs": matchs,
        "games": sans_features(existant["games"]),
        "joueurs": sans_features(existant["entries"]),
        "stats": resumer(historique, list(VERSIONS)),
        "cotes_quota": quota,
        "labo": {k: labo.get(k) for k in ("journal", "patterns", "donnees", "dernier_tuning")} | {
            "versions": [{k: v.get(k) for k in ("id", "cree", "desc", "retiree", "essai")} for v in labo.get("versions", [])]},
    })
    print(f"{jour}: {len(matchs)} matchs, {len(existant['entries'])} joueurs, cotes={'oui' if AVEC_COTES else 'non'}")


if __name__ == "__main__":
    main()
