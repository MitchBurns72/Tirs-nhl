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

import alignements
import auto
import cotes
import donnees_v2
import live
import nhl_api as api
from modeles import (DISPERSION, FAMILLES, SEUILS_TIRS, VERSIONS, ajouter_marches, ajouter_seuils, charger_auto,
                     features_joueur, features_match, p_seuil)
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
# Blessures ESPN + sondes DailyFaceoff (les tests le coupent : pas d'internet)
AVEC_ALIGNEMENTS = os.environ.get("ALIGNEMENTS", "1") == "1"
MARCHES_PROPS = [m for m in os.environ.get("ODDS_PROPS", "player_shots_on_goal_alternate").split(",") if m]
# config.json à la racine (livré dans la zip) : choisir les marchés de cotes sans toucher au workflow
try:
    _CONFIG = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text(encoding="utf-8"))
    MARCHES_PROPS = _CONFIG.get("odds_props") or MARCHES_PROPS
except (FileNotFoundError, ValueError):
    _CONFIG = {}


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
                      "mu_cur": gf_tot / gp_tot if gp_tot else None, "poids": poids, "feuilles": feuilles}


def construire_joueurs(jour, equipes, saison, ligue_eq, ligue, ctx_eq):
    prev_saison = saison_precedente(saison)
    joueurs = []
    for team in equipes:
        r = api.roster(team) or {}
        for grp in ("forwards", "defensemen"):
            for j in r.get(grp, []):
                joueurs.append({"pid": j["id"], "team": team, "pos": pos_de(j.get("positionCode")),
                                "name": f'{j["firstName"]["default"]} {j["lastName"]["default"]}',
                                # Pour la page : photo officielle NHL et numéro de chandail
                                "photo": j.get("headshot"), "no": j.get("sweaterNumber")})

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
                  "assists": g.get("assists", 0), "toi": toi_min(g.get("toi")), "opp": g["opponentAbbrev"],
                  "gid": str(g.get("gameId", ""))}
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
                     "vs": vs, "league": ligue, "gameId": info["gameId"], "incertain": incertain,
                     "eq_pj": len(dates)})
    return ctxs


# --- 4. Données v2 ----------------------------------------------------------------

def facteurs_rink(feuilles, jour, k=10):
    """Biais de patinoire, AVANT `jour` : total de tirs comptés dans les matchs à domicile
    d'une équipe ÷ total dans ses matchs à l'étranger. La qualité de l'équipe s'annule
    (c'est la même équipe des deux bords) ; ce qui reste, c'est le marqueur officiel de
    son aréna. Régressé vers 1 avec k matchs fictifs : 5 matchs ne prouvent rien."""
    par = {}
    for fb in feuilles.values():
        if fb["date"] >= jour or fb.get("hsog") is None or fb.get("asog") is None:
            continue
        tot = fb["hsog"] + fb["asog"]
        h, a = par.setdefault(fb["home"], [0, 0, 0, 0]), par.setdefault(fb["away"], [0, 0, 0, 0])
        h[0] += tot; h[1] += 1
        a[2] += tot; a[3] += 1
    out = {}
    for t, (th, nh, ta, na) in par.items():
        brut = (th / nh) / (ta / na) if nh and na and ta else 1.0
        out[t] = round((nh * brut + k) / (nh + k), 4)
    return out


def preparer_v2(jour, saison, feuilles):
    """Tout ce que la v2 sait d'une soirée, filtré AVANT `jour` (pas de fuite du futur) :
    stats par match (tentatives, PP), saison passée, blocs d'équipe, moyennes de ligue."""
    prev = saison_precedente(saison)
    jeux = donnees_v2.cache_stats_jeux(DIR_CACHE, saison, [g for g, fb in feuilles.items() if fb["date"] < jour])
    joueurs_prev = donnees_v2.cache_stats_saison(DIR_CACHE, prev)
    eq_prev = donnees_v2.equipes_saison_passee(DIR_CACHE, prev)

    par_eq = {}
    S = G = N = 0
    tous_pp = []
    for gid, fb in sorted(feuilles.items(), key=lambda kv: kv[1]["date"]):
        if fb["date"] >= jour:
            continue
        pp = (jeux.get(gid) or {}).get("pp", {})
        for moi, eux, sf, sa, gf, ga in ((fb["home"], fb["away"], fb["hsog"], fb["asog"], fb["hg"], fb["ag"]),
                                         (fb["away"], fb["home"], fb["asog"], fb["hsog"], fb["ag"], fb["hg"])):
            par_eq.setdefault(moi, []).append({"sf": sf, "sa": sa, "gf": gf, "ga": ga,
                                               "ppa": pp.get(eux)})   # minutes de PP données à l'adversaire
            S += sf; G += gf; N += 1
            if pp.get(eux) is not None:
                tous_pp.append(pp[eux])
    prevs = [v for v in eq_prev.values() if v]
    psf = sum(v[1] for v in prevs) / sum(v[0] for v in prevs) if prevs else 30.0
    psh = sum(v[3] for v in prevs) / sum(v[1] for v in prevs) if prevs else 0.095
    W = 480   # ≈ 15 matchs × 32 équipes de saison passée comme ancre
    lg_sf = (S + W * psf) / (N + W)
    lg_sh = (G + W * psf * psh) / (S + W * psf)
    lpp = (sum(tous_pp) + 300 * 4.5) / (len(tous_pp) + 300)   # ~4.5 min de PP par équipe par match

    def bloc(lst):
        return [sum(x["sf"] for x in lst), sum(x["sa"] for x in lst), sum(x["gf"] for x in lst),
                sum(x["ga"] for x in lst), len(lst)]
    blocs, osh = {}, {}
    for t, lst in par_eq.items():
        pv = eq_prev.get(t)
        blocs[t] = {"l10": bloc(lst[-10:]), "rs": bloc(lst[:-10]), "pv": pv}
        connus = [x["ppa"] for x in lst if x["ppa"] is not None]
        osh[t] = (sum(connus) + 10 * lpp) / (len(connus) + 10)   # régressé : 10 matchs à la moyenne
    return {"jeux": jeux, "prev": joueurs_prev, "blocs": blocs, "osh": osh, "lpp": lpp, "lg": [lg_sf, lg_sh],
            "rink": facteurs_rink(feuilles, jour)}


def features_x_joueur(ctx, v2, pw, ratio_pos):
    """Features v2 d'un joueur (voir modeles.py, section v2)."""
    stats = v2["jeux"]
    pid = str(ctx["pid"])
    def bloc(gs):
        att = toi_a = sog_a = pp = n_pp = 0
        for g in gs:
            st = ((stats.get(g.get("gid")) or {}).get("j") or {}).get(pid)
            if st and st[0] is not None:
                att += st[0]; toi_a += g["toi"]; sog_a += g["shots"]
            if st:
                pp += st[1]; n_pp += 1
        return att, toi_a, sog_a, pp, n_pp
    cur = ctx["cur"]
    a1, t1, s1, p1, n1 = bloc(cur[-10:])
    a2, t2, s2, p2, n2 = bloc(cur[:-10])
    pv = v2["prev"].get(pid)
    l5 = [((stats.get(g.get("gid")) or {}).get("j") or {}).get(pid) for g in cur[-5:]]
    l5 = [x[1] for x in l5 if x]
    return {
        "a": [a1, round(t1, 2), s1, a2, round(t2, 2), s2, pv[0] if pv else None, pv[2] if pv else 0],
        "pp": [round(p1, 2), n1, round(p2, 2), n2, pv[1] if pv else 0, pv[2] if pv else 0,
               round(sum(l5) / len(l5), 2) if l5 else None],
        "osh": v2["osh"].get(ctx["opp"]["abbrev"], v2["lpp"]), "lpp": v2["lpp"],
        "pw": pw, "lr": ratio_pos.get(ctx["pos"], 0.53),
        # Facteur de patinoire de l'aréna de CE SOIR (celle de l'équipe à domicile)
        "rk": v2.get("rink", {}).get(ctx["team"] if ctx["home"] else ctx["opp"]["abbrev"], 1.0),
    }


def implicites_joueur(cotes_j):
    """Cotes US d'un joueur -> probabilités implicites brutes (avec la marge du bookie)."""
    return {m: 1 / c["prix"] for m, c in (cotes_j or {}).items() if c and c.get("prix")}


def implicites_match(c):
    """Cotes US d'un match -> probabilités SANS marge (on a les deux côtés)."""
    if not c:
        return {}
    out = {}
    if c.get("p_home_marche"):
        out["home"] = c["p_home_marche"]
    lignes = dict(c.get("totaux") or {})
    if c.get("ligne") is not None and c.get("over") and c.get("under"):
        lignes.setdefault(str(c["ligne"]), {"over": c["over"], "under": c["under"]})
    for ligne, d in lignes.items():
        if ligne in ("5.5", "6.5") and d.get("over") and d.get("under"):
            io, iu = 1 / d["over"], 1 / d["under"]
            out["o" + ligne] = io / (io + iu)
    return out


def predire_version(spec, quoi, f, ctx=None, jeu=None, cm=None):
    """Une version « paramétrique » (v2, auto) se calcule à partir des features ;
    v0/v1 gardent leur code d'origine (gelé)."""
    if "params" in spec:
        return FAMILLES[spec.get("famille", "v1")][quoi](f, spec["params"])
    return spec[quoi](dict(ctx), jeu) if quoi == "joueur" else spec[quoi](cm)


# Une version de catégorie (tirs3_v5, vic_v6...) ne sert qu'à SON marché : on ne garde que
# celui-là (data.json reste léger et le Suivi ne la note pas ailleurs).
MARCHES_CAT = {"tirs1": ("tirs1",), "tirs2": ("tirs",), "tirs3": ("tirs3",), "tirs4": ("tirs4",),
               "but": ("buts",), "but2": ("buts2",), "but3": ("buts3",), "passe": ("passes",),
               "passe2": ("passes2",), "pts1": ("pts1",), "pts2": ("pts2",)}


def pertinent(spec, quoi):
    """Cette version prédit-elle les joueurs (quoi='joueur') ou les matchs ('match') ?"""
    cat = spec.get("cat")
    if not cat:
        return True
    return (cat in MARCHES_CAT) == (quoi == "joueur")


def elaguer(spec, out, quoi):
    cat = spec.get("cat")
    if not cat:
        return out
    if quoi == "joueur":
        garde = {m: out[m] for m in MARCHES_CAT.get(cat, ()) if m in out}
        detail = (out.get("tirs") or {}).get("detail")
        for m in garde:
            if m.startswith("tirs") and detail and "detail" not in garde[m]:
                garde[m] = dict(garde[m], detail=detail)   # pour « Pourquoi ce % »
        return garde
    if cat == "vic":
        return {k: out[k] for k in ("p_home", "p_home0", "p_egal60", "lam_home", "lam_away") if k in out}
    if cat == "tot":
        return {k: out[k] for k in ("p_over", "total", "lam_home", "lam_away") if k in out}
    return out


def appliquer_marche(entries, games):
    """Recalcule les versions de la famille v2 avec les cotes du marché, une fois
    les cotes connues. Les autres versions (v0, v1...) ne sont jamais touchées."""
    for g in games:
        mk = implicites_match(g.get("cotes"))
        if g.get("f") is None:
            continue
        g["f"]["mkt"] = mk or None
        for v, spec in VERSIONS.items():
            if spec.get("famille") == "v2" and v in g.get("preds", {}):
                g["preds"][v] = arrondir(elaguer(spec, predire_version(spec, "match", g["f"]), "match"))
    for e in entries:
        if e.get("f") is None:
            continue
        e["f"]["mkt"] = implicites_joueur(e.get("cotes")) or None
        for v, spec in VERSIONS.items():
            if spec.get("famille") == "v2" and v in e.get("preds", {}):
                pv = predire_version(spec, "joueur", e["f"])
                ajouter_seuils(pv, spec.get("nb_k"))
                e["preds"][v] = arrondir(elaguer(spec, pv, "joueur"))


# --- 5. Prédire ------------------------------------------------------------------

def predire(jour, matchs_ouverts, saison):
    equipes = {}
    for m in matchs_ouverts:
        equipes[m["away"]] = {"opp": m["home"], "home": False, "gameId": m["id"]}
        equipes[m["home"]] = {"opp": m["away"], "home": True, "gameId": m["id"]}

    ligue_eq, noms, ligue = contexte_ligue(jour, saison)
    ctx_eq, cache_prev = contexte_equipes(jour, equipes, saison)
    try:
        v2 = preparer_v2(jour, saison, ligue["feuilles"])
    except Exception as ex:  # noqa: BLE001
        # La v2 est un ajout : si l'API de stats flanche, v0/v1 roulent quand même
        print(f"  données v2 indisponibles : {str(ex)[:120]}")
        v2 = None

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

    # Toutes les versions prédisent, même un soir d'avant leur naissance (rétroactif) :
    # c'est le JUGEMENT qui écarte les soirs qui ont servi à ajuster leurs paramètres
    # (voir `juge_depuis` dans auto.py et score.py). Plus de data, sans tricher.
    actives = dict(VERSIONS)

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
        if v2:
            fm["x"] = {"h": v2["blocs"].get(m["home"]), "a": v2["blocs"].get(m["away"]), "lg": v2["lg"]}
            if not (fm["x"]["h"] and fm["x"]["a"]):
                fm.pop("x")
        for v, spec in actives.items():
            r = predire_version(spec, "match", fm, cm=cm)
            if pertinent(spec, "match"):
                preds[v] = elaguer(spec, r, "match")
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
    ctxs = construire_joueurs(jour, equipes, saison, ligue_eq, ligue, ctx_eq)
    # % de victoire de chaque équipe selon v1 (pour l'état de match de la v2)
    pw = {}
    for g in games:
        ph = g["preds"].get("v1", {}).get("p_home")
        if ph is not None:
            pw[g["home"]], pw[g["away"]] = ph, 1 - ph
    # Ratio tirs cadrés / tentatives de chaque position, sur le pool du soir
    ratio_pos = {}
    if v2:
        for pos in ("F", "D"):
            sog = att = 0
            for c in ctxs:
                if c["pos"] != pos:
                    continue
                for gm in c["cur"]:
                    st = ((v2["jeux"].get(gm.get("gid")) or {}).get("j") or {}).get(str(c["pid"]))
                    if st and st[0]:
                        sog += gm["shots"]; att += st[0]
            ratio_pos[pos] = sog / att if att > 200 else 0.53
    for ctx in ctxs:
        # Features calculées avec les buts attendus de v1 : c'est la référence
        # commune sur laquelle le tuner rejoue toutes les variantes.
        f = features_joueur(ctx, jeu_par_equipe.get("v1", {}).get(ctx["team"]))
        if v2:
            f["x"] = features_x_joueur(ctx, v2, pw.get(ctx["team"]), ratio_pos)
        # La série de tirs du joueur (dispersion propre à chaque joueur)
        f["s10"] = [g["shots"] for g in ctx["cur"][-10:]]
        joueurs_v = {v: spec for v, spec in actives.items() if pertinent(spec, "joueur")}
        preds = {v: predire_version(spec, "joueur", f, ctx=ctx, jeu=jeu_par_equipe[v].get(ctx["team"]))
                 for v, spec in joueurs_v.items()}
        for v, spec in joueurs_v.items():
            ajouter_seuils(preds[v], spec.get("nb_k"))   # tirs 1+, 3+, 4+
            preds[v] = elaguer(spec, preds[v], "joueur")
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
            "photo": ctx.get("photo"), "no": ctx.get("no"), "saison": saison_joueur(ctx),
            "preds": arrondir(preds), "result": None, "played": None,
            "f": arrondir(f),
        })
    return games, entries, noms


def saison_joueur(ctx, n_matchs=82):
    """Stats de la saison (avant ce soir) + projection de fin de saison. La projection
    part de SON rythme par match joué, sur les matchs qui RESTENT à son équipe : un gars
    qui a manqué la moitié de la saison ne récupère pas ses matchs perdus."""
    cur = ctx["cur"]
    pj = len(cur)
    tot = {"b": sum(g["goals"] for g in cur), "a": sum(g["assists"] for g in cur), "tirs": sum(g["shots"] for g in cur)}
    tot["pts"] = tot["b"] + tot["a"]
    restants = max(0, n_matchs - ctx.get("eq_pj", pj))
    out = {"pj": pj, **tot, "toi": round(sum(g["toi"] for g in cur) / pj, 1) if pj else None,
           "eq_pj": ctx.get("eq_pj"), "restants": restants}
    if pj:
        out["proj"] = {k: round(v + v / pj * restants) for k, v in tot.items()}
        out["max_pj"] = pj + restants
    return out


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
MAX_COMPLETER = 60   # fichiers complétés par exécution


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
        # Toutes les versions paramétriques, auto comprises : elles se rejouent à partir des
        # features, sans réseau. Une prédiction déjà là n'est jamais recalculée.
        a_rejouer = [v for v, sp in VERSIONS.items() if "params" in sp]
        marque = "x|rk|" + ",".join(a_rejouer)
        # Facteur de patinoire ajouté le 23 octobre : on le colle aux vieilles lignes
        # (calculé avec les feuilles d'AVANT ce soir-là, donc pas de triche) pour que le
        # tuner puisse apprendre s'il sert. Aucune prédiction n'est modifiée.
        sans_rk = [e for e in data["entries"] if (e.get("f") or {}).get("x") and "rk" not in e["f"]["x"]]
        if sans_rk:
            rk = facteurs_rink(lire_json(DIR_CACHE / f"feuilles_{saison}.json", {}), data["date"])
            for e in sans_rk:
                e["f"]["x"]["rk"] = rk.get(e["team"] if e["home"] else e["opp"], 1.0)
        if data.get("complete") == marque:
            continue   # déjà fait : un joueur échangé depuis n'aura jamais de features, on n'insiste pas
        # Il manque quelque chose si : pas de features, pas de features v2, ou une
        # version paramétrique non-auto (v2) absente des prédictions.
        r_j = [v for v in a_rejouer if pertinent(VERSIONS[v], "joueur")]
        r_g = [v for v in a_rejouer if pertinent(VERSIONS[v], "match")]
        manque = lambda x, rv: not x.get("f") or "x" not in x["f"] or any(v not in x.get("preds", {}) for v in rv)
        manque_j = [e for e in data["entries"] if manque(e, r_j)]
        manque_g = [g for g in data.get("games", []) if manque(g, r_g)]
        for e in data["entries"]:
            if e.get("f") and "s10" not in e["f"] and e.get("l10_shots") is not None:
                e["f"]["s10"] = e["l10_shots"]
        if not manque_j and not manque_g:
            data["complete"] = marque
            ecrire_json(fic, data)
            continue
        # Les features manquent seulement sur les très vieux soirs. Si toutes les lignes
        # ont déjà leurs features, on rejoue les versions manquantes SANS aucun appel
        # réseau (c'est le cas quand on ajoute v2.1 à v2.5 : tout est déjà là).
        sans_f = lambda x: not x.get("f") or "x" not in x["f"]
        if any(sans_f(x) for x in manque_j + manque_g):
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
            # On garde TOUJOURS les features d'origine quand elles existent (calculées avant le match)
            for e in manque_j:
                if sans_f(e):
                    e["f"] = f_j.get((e["gameId"], e["pid"])) or e.get("f")
            for g in manque_g:
                if sans_f(g):
                    g["f"] = f_g.get(g["gameId"]) or g.get("f")
        # Rejouer les versions manuelles (paramètres fixés d'avance, pas des versions auto)
        # qui manquent, avec les cotes d'avant-match si on les avait. Une prédiction déjà
        # là n'est jamais recalculée, et les autres versions ne sont jamais touchées.
        for x in manque_g:
            if x.get("f") is not None:
                if "mkt" not in x["f"]:
                    x["f"]["mkt"] = implicites_match(x.get("cotes")) or None
                for v in r_g:
                    if v not in x.setdefault("preds", {}):
                        x["preds"][v] = arrondir(elaguer(VERSIONS[v], predire_version(VERSIONS[v], "match", x["f"]), "match"))
        for x in manque_j:
            if x.get("f") is not None:
                if "mkt" not in x["f"]:
                    x["f"]["mkt"] = implicites_joueur(x.get("cotes")) or None
                for v in r_j:
                    if v not in x.setdefault("preds", {}):
                        pv = predire_version(VERSIONS[v], "joueur", x["f"])
                        ajouter_seuils(pv, VERSIONS[v].get("nb_k"))
                        x["preds"][v] = arrondir(elaguer(VERSIONS[v], pv, "joueur"))
        data["complete"] = marque
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
                if v in DISPERSION and "_v" not in v and ajouter_seuils(pv, DISPERSION[v]):
                    modifie = True
        if modifie:
            ecrire_json(fic, data)


def archiver_soirs(jour, n=3, garder=21):
    """Écrit docs/live/AAAA-MM-JJ.json (score + stats finales de chaque joueur)
    pour les n derniers soirs, dans le même format que live.json.

    POURQUOI : l'onglet Bets de la page règle tes paris des soirs passés avec ces
    fichiers (live.json ne contient que la soirée en cours). Un soir est réécrit
    tant que tous ses matchs ne sont pas finis. Les archives de plus de `garder`
    jours sont effacées : un bet réglé garde son résultat sur ton appareil."""
    dossier = SORTIE.parent / "live"
    dossier.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        d = (date.fromisoformat(jour) - timedelta(days=i)).isoformat()
        f = dossier / f"{d}.json"
        if (lire_json(f, {}) or {}).get("complet"):
            continue
        sched = api.schedule(d) or {}
        jeux = [g for g in next((x["games"] for x in sched.get("gameWeek", []) if x["date"] == d), [])
                if g.get("gameType") in (2, None)]
        if not jeux:
            continue
        games = {}
        for g in jeux:
            b = api.boxscore(g["id"])
            if b and b.get("gameState") not in ETATS_AVANT:
                games[str(g["id"])] = live.etat_match(b)
        complet = len(games) == len(jeux) and all(x["state"] in ETATS_FINIS for x in games.values())
        f.write_text(json.dumps({"date": d, "complet": complet, "games": games,
                                 "maj": datetime.now(timezone.utc).isoformat(timespec="minutes")},
                                ensure_ascii=False), encoding="utf-8")
    limite = (date.fromisoformat(jour) - timedelta(days=garder)).isoformat()
    for f in dossier.glob("*.json"):
        if f.stem < limite:
            f.unlink()


# La version « combo » : pour chaque marché, le % du champion de sa catégorie.
CAT_DE = {"tirs1": "tirs1", "tirs": "tirs2", "tirs3": "tirs3", "tirs4": "tirs4", "buts": "but", "buts2": "but2",
          "buts3": "but3", "passes": "passe", "passes2": "passe2", "pts1": "pts1", "pts2": "pts2"}
# Les seuils d'un même stat doivent rester dans l'ordre (1+ >= 2+ >= 3+...)
ORDRES = [("tirs1", "tirs", "tirs3", "tirs4"), ("buts", "buts2", "buts3"), ("passes", "passes2"), ("pts1", "pts2")]
COMBO = "combo"


def ajouter_combo(entries, games, principales):
    """Ajoute preds["combo"] en COPIANT les % des champions de chaque catégorie (choisis
    avec les données d'avant ce soir). "src" dit d'où vient chaque chiffre.
    Garde-fou : chaque seuil de tirs a sa propre recette, donc on force l'ordre
    1+ >= 2+ >= 3+ >= 4+ (sinon un gars aurait plus de chances de faire 3 tirs que 2)."""
    if not principales:
        return
    for e in entries:
        combo = {}
        for m, cat in CAT_DE.items():
            pv = e["preds"].get(principales.get(cat), {}).get(m)
            if pv:
                combo[m] = {**pv, "src": principales[cat]}
        for ordre in ORDRES:
            precedent = 1.0
            for m in ordre:
                if m in combo:
                    if combo[m]["p"] > precedent:
                        combo[m] = dict(combo[m], p=precedent, borne=True)
                    precedent = combo[m]["p"]
        if combo:
            e["preds"][COMBO] = combo
    for g in games:
        tot = g["preds"].get(principales.get("tot"))
        vic = g["preds"].get(principales.get("vic"))
        if not tot or not vic:
            continue
        # Buts attendus et totaux : champion des totaux ; % de victoire : champion de la victoire
        combo = {**tot, "src_tot": principales["tot"], "src_vic": principales["vic"]}
        for k in ("p_home", "p_home0"):
            if k in vic:
                combo[k] = vic[k]
            else:
                combo.pop(k, None)
        g["preds"][COMBO] = combo


# % du soir : mêmes formules que la page (docs/index.html, soirJoueur)
TOI_TRIO = {"f1": 19.0, "f2": 16.5, "f3": 14.0, "f4": 11.0, "d1": 23.0, "d2": 20.0, "d3": 17.0}
PP_UNITE = {"pp1": 2.9, "pp2": 1.3, "aucune": 0.15}
SV_LIGUE = 0.903
SOIR = "soir"


def ajouter_soir(entries, align, vp):
    """preds["soir"] = la version principale ajustée à l'alignement DailyFaceoff :
    temps de glace (moitié projeté, moitié typique du trio), unité de PP, gardien adverse.
    Poids fixés d'avance : c'est le Suivi qui dira si ça aide. Retourne le nombre ajusté."""
    from math import exp, sqrt
    n = 0
    for e in entries:
        al = (align.get("equipes") or {}).get(e["team"])
        base = e["preds"].get(vp) or {}
        if not al or "tirs" not in base:
            continue
        nom = next((x for x in al["joueurs"] if cotes.norm(x) == cotes.norm(e["name"])), None)
        dans = lambda g: any(cotes.norm(x) == cotes.norm(e["name"]) for x in al["groupes"].get(g, []))
        slot = next((g for g in TOI_TRIO if dans(g)), None)
        if not nom or not slot or (al["joueurs"].get(nom) or {}).get("inj"):
            e["soir_info"] = {"absent": True}
            continue
        unite = "pp1" if dans("pp1") else "pp2" if dans("pp2") else "aucune"
        dets = [pv.get("tirs", {}).get("detail") for pv in e["preds"].values() if isinstance(pv, dict)]
        dets = [d for d in dets if d]
        toi = next((d["toi"] for d in dets if d.get("toi")), None)
        pp = next((d["pp"] for d in dets if d.get("pp") is not None), None)
        if not toi:
            continue
        toi_s = 0.5 * toi + 0.5 * TOI_TRIO[slot]
        f_toi = toi_s / toi
        d_pp = PP_UNITE[unite] - pp if pp is not None else 0.0
        borne = lambda x: min(1.5, max(0.6, x))
        f_t, f_a = borne(f_toi * (1 + d_pp / toi_s)), borne(f_toi * (1 + 1.5 * d_pp / toi_s))
        g = (align.get("gardiens") or {}).get(e["opp"]) or {}
        try:
            sv = float(g.get("sv")) if g.get("sv") else None
        except ValueError:
            sv = None
        f_g = min(1.15, max(0.85, sqrt((1 - sv) / (1 - SV_LIGUE)))) if sv else 1.0
        soir = {}
        lt = base["tirs"]["lam"] * f_t
        for m, seuil in SEUILS_TIRS.items():
            soir[m] = {"p": p_seuil(lt, seuil, 15.0), "lam": lt}
        if "buts" in base:
            lb = base["buts"]["lam"] * f_t * f_g
            soir["buts"] = {"p": 1 - exp(-lb), "lam": lb}
        if "passes" in base:
            la = base["passes"]["lam"] * f_a
            soir["passes"] = {"p": 1 - exp(-la), "lam": la}
        ajouter_marches(soir)
        e["preds"][SOIR] = arrondir(soir)
        e["soir_info"] = arrondir({"slot": slot, "pp": unite, "f_t": f_t, "f_a": f_a, "f_g": f_g, "base": vp})
        n += 1
    return n


def sans_features(liste):
    """La page n'a pas besoin des features (lourdes) : on les retire de data.json."""
    return [{k: v for k, v in x.items() if k != "f"} for x in liste]


def _r3(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _r3(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r3(v) for v in x]
    return x


def allege(entries):
    """data.json pour le cell : avec 8-10 versions, chaque joueur traîne beaucoup de
    chiffres. On arrondit à 3 décimales (0.1 % près, plus que ce que la page affiche)
    et on retire le λ des tirs 1+/3+/4+ (c'est le même que celui des tirs 2+, la page
    le reprend). Les fichiers de data/predictions gardent TOUT, au complet."""
    out = []
    for e in sans_features(entries):
        preds = {}
        for v, pv in (e.get("preds") or {}).items():
            pv = {m: ({k: x for k, x in o.items() if k != "lam"} if m in ("tirs1", "tirs3", "tirs4") and isinstance(o, dict) else o)
                  for m, o in pv.items()}
            preds[v] = pv
        out.append(_r3({**e, "preds": preds}))
    return out


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
    archiver_soirs(jour)   # résultats des derniers soirs pour régler tes bets
    if AVEC_ALIGNEMENTS:
        try:
            alignements.sonder(RACINE / "data" / "sondes", jour)
        except Exception as ex:  # noqa: BLE001
            print(f"  sondes : {str(ex)[:120]}")

    # Labo : patterns + tuning hebdo + promotion. Si une nouvelle version naît,
    # elle est chargée tout de suite et prédit déjà les matchs de ce soir.
    historique = [h for h in (lire_json(f, None) for f in sorted(DIR_PRED.glob("*.json"))) if h]
    labo, nouvelle = auto.cycle(historique, ETAT_AUTO, list(VERSIONS), jour)
    # Toujours recharger : le cycle a pu faire naître des versions, en retirer, ou
    # changer leurs dates de jugement (juge_depuis).
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

    # Alignements DailyFaceoff (trios, PP, gardiens) des équipes de ce soir. Le job live
    # les rafraîchit aux 30 min dans docs/live.json ; la page prend le plus récent.
    align = None
    if AVEC_ALIGNEMENTS and matchs:
        try:
            align = alignements.alignements_soir([t for m in matchs for t in (m["home"], m["away"])])
            print(f"  alignements : {len(align['equipes'])} équipes, {len(align['gardiens'])} gardiens")
        except Exception as ex:  # noqa: BLE001
            print(f"  alignements : {str(ex)[:120]}")

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
        # v2 : mélange avec le marché maintenant que les cotes sont connues
        appliquer_marche(entries, games)
        # Combo : le champion de chaque catégorie (ce soir et les matchs de ce soir déjà partis)
        ajouter_combo(entries + entries_retro, games + games_retro, labo.get("principales"))
        # % du soir : la principale ajustée à l'alignement (noté au Suivi comme une version)
        if align:
            n_s = ajouter_soir(entries, align, COMBO if labo.get("principales") else principale)
            print(f"  % du soir : {n_s} joueurs ajustés à l'alignement")
        # Blessures du jour (ESPN) : on marque, on ne retire pas
        if AVEC_ALIGNEMENTS and entries:
            n_b = alignements.annoter(entries, alignements.blessures())
            print(f"  blessures ESPN : {n_b} joueurs de ce soir sur la liste")
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
        if not ouverts:
            cotes.tester_cle()   # aucun match ouvert : on vérifie juste que la clé marche
        quota = {**cotes.etat, "quand": datetime.now(timezone.utc).isoformat(timespec="minutes")}

    ecrire_json(SORTIE, {
        "genere": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "date": jour,
        "versions": [{"id": COMBO, "desc": "Le champion de chaque catégorie : " + ", ".join(
                         f"{c} {v}" for c, v in (labo.get("principales") or {}).items())},
                     {"id": SOIR, "desc": "La principale ajustée à l'alignement DailyFaceoff du soir : temps de glace "
                                          "selon le trio, unité de PP, gardien adverse (poids fixés d'avance)"}] +
                    [{"id": v, "desc": s["desc"], "cree": s.get("cree"), "juge_depuis": s.get("juge_depuis"),
                      "cat": s.get("cat")} for v, s in VERSIONS.items()],
        "principale": COMBO if labo.get("principales") else principale,
        "principales": labo.get("principales"),
        "alignements": align,
        "matchs": matchs,
        "games": sans_features(existant["games"]),
        "joueurs": allege(existant["entries"]),
        "stats": resumer(historique, [COMBO, SOIR] + list(VERSIONS), {v: s.get("juge_depuis") for v, s in VERSIONS.items()}),
        "cotes_quota": quota,
        "labo": {k: labo.get(k) for k in ("journal", "patterns", "donnees", "principales", "recalcul",
                                          "prochain_recalcul", "dernier_recalcul")} | {
            "versions": [{k: v.get(k) for k in ("id", "cat", "cree", "desc", "retiree", "base", "rapport")}
                         for v in labo.get("versions", []) if v.get("cat")]},
    })
    print(f"{jour}: {len(matchs)} matchs, {len(existant['entries'])} joueurs, cotes={'oui' if AVEC_COTES else 'non'}")


if __name__ == "__main__":
    main()
