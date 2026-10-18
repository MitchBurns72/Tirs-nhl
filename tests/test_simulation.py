"""
Test de bout en bout SANS internet : on remplace l'API NHL ET The Odds API par une
fausse ligue.

POURQUOI : valider tout le pipeline (prédire -> grader -> scorer -> data.json)
avant de pousser une mise à jour. La fausse ligue a un vrai « talent » caché
(taux de tirs, % de tir, force d'attaque/défense des équipes), donc on peut
vérifier que v1 se débrouille au moins aussi bien que la baseline v0.

Lancement : python tests/test_simulation.py
"""

import json
import math
import random
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "src"))

import cotes  # noqa: E402
import nhl_api as api  # noqa: E402
import run  # noqa: E402

rng = random.Random(42)
EQUIPES = ["MTL", "TOR", "BOS", "DET", "NYR", "PHI", "PIT", "WSH"]
NOMS = {"MTL": "Canadiens", "TOR": "Maple Leafs", "BOS": "Bruins", "DET": "Red Wings",
        "NYR": "Rangers", "PHI": "Flyers", "PIT": "Penguins", "WSH": "Capitals"}
SAISON, PREV = 20262027, 20252026
DEBUT = date(2026, 10, 7)

joueurs, pid = {}, 8470000
for t in EQUIPES:
    for i in range(18):
        pid += 1
        d = i >= 12
        joueurs[pid] = {"team": t, "pos": "D" if d else "C",
                        "rate": rng.uniform(0.05, 0.12) if d else rng.uniform(0.06, 0.22),
                        "toi": rng.uniform(17, 24) if d else rng.uniform(10, 20),
                        "playmaker": rng.uniform(0.5, 2.0)}
attaque = {t: rng.uniform(0.85, 1.15) for t in EQUIPES}
defense = {t: rng.uniform(0.85, 1.15) for t in EQUIPES}


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


calendrier = {}
for n in range(14):
    d = (DEBUT + timedelta(days=n)).isoformat()
    t = EQUIPES[:]
    rng.shuffle(t)
    calendrier[d] = [(2026020000 + n * 10 + i, t[2 * i], t[2 * i + 1]) for i in range(4)]

logs = {p: [] for p in joueurs}
box, etat, scores = {}, {}, {}


def jouer(d, matchs=None, cible_logs=None, cible_box=None, cible_scores=None, cible_etat=None):
    cible_logs = logs if cible_logs is None else cible_logs
    cible_box = box if cible_box is None else cible_box
    cible_scores = scores if cible_scores is None else cible_scores
    cible_etat = etat if cible_etat is None else cible_etat
    for gid, away, home in (matchs or calendrier[d]):
        stats = {"awayTeam": {"forwards": [], "defense": []}, "homeTeam": {"forwards": [], "defense": []}}
        lignes = {}
        for p, j in joueurs.items():
            if j["team"] not in (away, home) or rng.random() < 0.05:
                continue
            opp = home if j["team"] == away else away
            toi = max(5, rng.gauss(j["toi"], 2.5))
            s = poisson(j["rate"] * toi * defense[opp])
            lignes[p] = {"team": j["team"], "opp": opp, "toi": toi, "shots": s, "goals": 0, "assists": 0}
        # Buts : chaque tir rentre à ~9.5 % x qualité du gardien adverse
        for p, l in lignes.items():
            l["goals"] = sum(1 for _ in range(l["shots"]) if rng.random() < 0.095 * defense[l["opp"]] * attaque[l["team"]])
        for p, l in lignes.items():
            for _ in range(l["goals"]):
                coeq = [q for q in lignes if lignes[q]["team"] == l["team"] and q != p]
                poids = [joueurs[q]["playmaker"] * lignes[q]["toi"] for q in coeq]
                for q in set(rng.choices(coeq, poids, k=2)):
                    lignes[q]["assists"] += 1
        sc = {away: 0, home: 0}
        for p, l in lignes.items():
            sc[l["team"]] += l["goals"]
            cible_logs.setdefault(p, []).append({"gameId": gid, "teamAbbrev": l["team"], "gameDate": d, "shots": l["shots"],
                            "goals": l["goals"], "assists": l["assists"],
                            "toi": f"{int(l['toi'])}:{int((l['toi'] % 1) * 60):02d}", "opponentAbbrev": l["opp"]})
            cote = "awayTeam" if l["team"] == away else "homeTeam"
            stats[cote]["defense" if joueurs[p]["pos"] == "D" else "forwards"].append(
                {"playerId": p, "sog": l["shots"], "goals": l["goals"], "assists": l["assists"]})
        if sc[away] == sc[home]:
            sc[home if rng.random() < 0.52 else away] += 1   # prolongation/fusillade
        cible_scores[gid] = sc
        cible_box[gid] = {"gameState": "OFF", "gameDate": d, "playerByGameStats": stats,
                    "awayTeam": {"abbrev": away, "score": sc[away]}, "homeTeam": {"abbrev": home, "score": sc[home]}}
        cible_etat[gid] = "OFF"


# Saison passée : 41 rondes simulées avec le MÊME moteur (talents identiques)
logs_prev, box_prev, scores_prev, etat_prev, cal_prev = {}, {}, {}, {}, []
for n in range(41):
    t = EQUIPES[:]
    rng.shuffle(t)
    ms = [(2025020000 + n * 10 + i, t[2 * i], t[2 * i + 1]) for i in range(4)]
    cal_prev += ms
    jouer("2026-01-01", ms, logs_prev, box_prev, scores_prev, etat_prev)


def jeux_finis(t):
    return [(d, g) for d in calendrier for g in calendrier[d] if t in g[1:] and etat.get(g[0]) == "OFF"]


def faux_get_json(path, retries=4):
    parts = path.strip("/").split("/")
    if parts[0] == "schedule":
        d = parts[1]
        return {"gameWeek": [{"date": d, "games": [
            {"id": g, "gameType": 2, "season": SAISON, "startTimeUTC": f"{d}T23:00:00Z",
             "gameState": etat.get(g, "FUT"), "awayTeam": {"abbrev": a}, "homeTeam": {"abbrev": h}}
            for g, a, h in calendrier.get(d, [])]}]}
    if parts[0] == "roster":
        t = parts[1]
        mk = lambda p, j: {"id": p, "firstName": {"default": "Joueur"}, "lastName": {"default": str(p)[-3:]},
                           "positionCode": j["pos"]}
        return {"forwards": [mk(p, j) for p, j in joueurs.items() if j["team"] == t and j["pos"] != "D"],
                "defensemen": [mk(p, j) for p, j in joueurs.items() if j["team"] == t and j["pos"] == "D"],
                "goalies": []}
    if parts[0] == "player":
        p, saison = int(parts[1]), int(parts[3])
        if saison == SAISON:
            return {"gameLog": list(reversed(logs[p]))}
        return {"gameLog": logs_prev.get(p, [])}
    if parts[0] == "club-stats":
        t = parts[1]
        prev = len(parts) > 3
        bx = box_prev if prev else box
        ids = [g for g, a, h in cal_prev if t in (a, h)] if prev else [g for _, (g, a, h) in jeux_finis(t)]
        sa = ga = 0
        for g in ids:
            cote_opp = "awayTeam" if bx[g]["homeTeam"]["abbrev"] == t else "homeTeam"
            st = bx[g]["playerByGameStats"][cote_opp]
            sa += sum(x["sog"] for grp in st.values() for x in grp)
            ga += sum(x["goals"] for grp in st.values() for x in grp)
        if prev:   # run.py divise par 82 : on ramène le total à l'échelle de 82 matchs
            sa, ga = sa * 82 // max(len(ids), 1), ga * 82 // max(len(ids), 1)
        return {"goalies": [{"shotsAgainst": sa, "goalsAgainst": ga}]}
    if parts[0] == "club-schedule-season":
        t = parts[1]
        if parts[2] != "now":
            return {"games": [{"gameType": 2, "gameState": "OFF", "gameDate": "2026-01-01",
                               "homeTeam": {"abbrev": h, "score": scores_prev[g][h]},
                               "awayTeam": {"abbrev": a, "score": scores_prev[g][a]}}
                              for g, a, h in cal_prev if t in (a, h)]}
        games = []
        for d, (gid, a, h) in jeux_finis(t):
            games.append({"id": gid, "startTimeUTC": f"{d}T23:00:00Z", "gameType": 2, "gameState": "OFF", "gameDate": d,
                          "homeTeam": {"abbrev": h, "score": scores[gid][h]},
                          "awayTeam": {"abbrev": a, "score": scores[gid][a]}})
        return {"games": games}
    if parts[0] == "standings":
        return {"standings": [{"teamAbbrev": {"default": t}, "teamCommonName": {"default": NOMS[t]},
                               "gamesPlayed": len(jeux_finis(t)),
                               "goalFor": sum(scores[g[0]][t] for _, g in jeux_finis(t))} for t in EQUIPES]}
    if parts[0] == "gamecenter":
        return box.get(int(parts[1]), {"gameState": "FUT"})
    raise ValueError(path)


def faux_odds(path, params):
    """Imite les réponses de The Odds API (format décimal)."""
    d = jour_courant[0]
    evs = [{"id": f"ev{g}", "home_team": f"Ville {NOMS[h]}", "away_team": f"Ville {NOMS[a]}",
            "bookmakers": [{"key": "bk", "markets": [
                {"key": "h2h", "outcomes": [{"name": f"Ville {NOMS[h]}", "price": 1.85},
                                            {"name": f"Ville {NOMS[a]}", "price": 2.0}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 6.5, "price": 1.95},
                                               {"name": "Under", "point": 6.5, "price": 1.87}]}]}]}
           for g, a, h in calendrier[d]]
    if path == "/odds":
        return evs
    gid = int(path.split("/")[2][2:])
    _, a, h = next(x for x in calendrier[d] if x[0] == gid)
    outs = [{"name": "Over", "description": f"Joueur {str(p)[-3:]}", "point": 1.5, "price": 1.45}
            for p, j in joueurs.items() if j["team"] in (a, h)]
    return {"bookmakers": [{"key": "bk", "markets": [{"key": "player_shots_on_goal_alternate", "outcomes": outs}]}]}


jour_courant = [None]


def main():
    tmp = Path(tempfile.mkdtemp())
    run.DIR_PRED = tmp / "data" / "predictions"
    run.DIR_CACHE = tmp / "data" / "cache"
    run.SORTIE = tmp / "docs" / "data.json"
    api.get_json = faux_get_json
    cotes._get = faux_odds
    run.AVEC_COTES = True
    import os
    os.environ["ODDS_API_KEY"] = "test"

    jours = sorted(calendrier)
    for d in jours:
        api._cache.clear()
        jour_courant[0] = d
        sys.argv = ["run.py", d]
        run.main()
        if d != jours[-1]:
            jouer(d)

    # --- Rétroactif : 3 soirs « manqués » doivent être rejoués et gradés ---
    effaces = jours[4:7]
    for d in effaces:
        (run.DIR_PRED / f"{d}.json").unlink()
    api._cache.clear()
    sys.argv = ["run.py", jours[-1]]
    run.main()
    for d in effaces:
        f = json.loads((run.DIR_PRED / f"{d}.json").read_text())
        assert f.get("retro") is True, f"{d} pas marqué rétroactif"
        assert any(e.get("played") for e in f["entries"]), f"{d} pas gradé"
        assert all(g.get("result") for g in f["games"]), f"{d} matchs pas gradés"
    print("rétroactif OK :", effaces)

    out = json.loads(run.SORTIE.read_text())
    st = out["stats"]
    for m, par_v in list(st["joueurs"].items()) + list(st["matchs"].items()):
        print(f"{m:9}", {v: (s["n"], round(s["note"], 1), round(s["brier"], 4)) for v, s in par_v.items() if s["n"]})
    print("global  ", {v: round(s["note"], 1) for v, s in st["global"].items()})
    assert st["joueurs"]["tirs"]["v1"]["n"] > 300
    assert st["matchs"]["victoire"]["v1"]["n"] > 40
    assert out["games"] and out["joueurs"]
    assert out["games"][0].get("cotes", {}).get("p_home_marche"), "cotes de match non associées"
    assert any(e.get("cotes") for e in out["joueurs"]), "cotes joueurs non associées"
    dest = ICI / "data_exemple.json"
    shutil.copy(run.SORTIE, dest)
    print("OK ->", dest)


if __name__ == "__main__":
    main()
