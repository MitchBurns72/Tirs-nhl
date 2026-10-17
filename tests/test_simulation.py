"""
Test de bout en bout SANS internet : on remplace l'API NHL par une fausse ligue.

POURQUOI : on veut valider tout le pipeline (prédire -> grader -> scorer ->
data.json) avant de le brancher sur la vraie API. La fausse ligue a des joueurs
avec un vrai "talent" caché (taux de tirs), donc on peut vérifier que v1
se débrouille au moins aussi bien que la baseline v0.

Lancement : python tests/test_simulation.py
"""

import json
import random
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "src"))

import nhl_api as api  # noqa: E402
import run  # noqa: E402

rng = random.Random(42)
EQUIPES = ["MTL", "TOR", "BOS", "DET", "NYR", "PHI", "PIT", "WSH"]
SAISON, PREV = 20262027, 20252026
DEBUT = date(2026, 10, 7)

# Talent caché : tirs/min et TOI moyen par joueur; tirs accordés par équipe
joueurs = {}
pid = 8470000
for t in EQUIPES:
    for i in range(18):
        pid += 1
        d = i >= 12
        joueurs[pid] = {"team": t, "pos": "D" if d else "C",
                        "rate": rng.uniform(0.05, 0.12) if d else rng.uniform(0.06, 0.22),
                        "toi": rng.uniform(17, 24) if d else rng.uniform(10, 20)}
defense = {t: rng.uniform(0.85, 1.15) for t in EQUIPES}


def poisson(lam):
    # Knuth : suffisant pour des petits lambdas
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


# Calendrier : chaque jour, 4 matchs (équipes mélangées)
calendrier = {}
for n in range(12):
    d = (DEBUT + timedelta(days=n)).isoformat()
    t = EQUIPES[:]
    rng.shuffle(t)
    calendrier[d] = [(2026020000 + n * 10 + i, t[2 * i], t[2 * i + 1]) for i in range(4)]

logs = {p: [] for p in joueurs}        # game logs saison courante (remplis au fil des jours)
box = {}                               # feuilles de match
etat = {}                              # gameId -> état


def jouer(d):
    """Simule les matchs du jour d : remplit logs + boxscores."""
    for gid, away, home in calendrier[d]:
        stats = {"awayTeam": {"forwards": [], "defense": []}, "homeTeam": {"forwards": [], "defense": []}}
        for p, j in joueurs.items():
            if j["team"] not in (away, home):
                continue
            if rng.random() < 0.05:
                continue  # scratché ce soir
            opp = home if j["team"] == away else away
            toi = max(5, rng.gauss(j["toi"], 2.5))
            s = poisson(j["rate"] * toi * defense[opp])
            logs[p].append({"gameId": gid, "teamAbbrev": j["team"], "homeRoadFlag": "H" if j["team"] == home else "R",
                            "gameDate": d, "shots": s, "toi": f"{int(toi)}:{int((toi % 1) * 60):02d}",
                            "opponentAbbrev": opp})
            cote = "awayTeam" if j["team"] == away else "homeTeam"
            stats[cote]["defense" if j["pos"] == "D" else "forwards"].append({"playerId": p, "sog": s})
        box[gid] = {"gameState": "OFF", "playerByGameStats": stats}
        etat[gid] = "OFF"


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
        fw = [{"id": p, "firstName": {"default": "Joueur"}, "lastName": {"default": str(p)[-3:]},
               "positionCode": j["pos"]} for p, j in joueurs.items() if j["team"] == t and j["pos"] != "D"]
        df = [{"id": p, "firstName": {"default": "Défenseur"}, "lastName": {"default": str(p)[-3:]},
               "positionCode": "D"} for p, j in joueurs.items() if j["team"] == t and j["pos"] == "D"]
        return {"forwards": fw, "defensemen": df, "goalies": []}
    if parts[0] == "player":
        p, saison = int(parts[1]), int(parts[3])
        if saison == SAISON:
            return {"gameLog": list(reversed(logs[p]))}  # l'API donne le plus récent en premier
        # Saison passée : 60 matchs générés du talent caché
        j = joueurs[p]
        r = random.Random(p)
        gl = []
        for _ in range(60):
            toi = j["toi"]
            opp = r.choice([t for t in EQUIPES if t != j["team"]])
            gl.append({"gameDate": "2026-01-01", "shots": int(r.gauss(j["rate"] * toi, 1.2) + 0.5) if j["rate"] * toi > 0 else 0,
                       "toi": f"{int(toi)}:00", "opponentAbbrev": opp, "teamAbbrev": j["team"]})
        for g in gl:
            g["shots"] = max(0, g["shots"])
        return {"gameLog": gl}
    if parts[0] == "club-stats":
        t = parts[1]
        if len(parts) > 3:  # saison passée
            return {"goalies": [{"shotsAgainst": int(28 * defense[t] * 82)}]}
        gp = sum(1 for d in calendrier for g in calendrier[d] if t in g[1:] and etat.get(g[0]) == "OFF")
        return {"goalies": [{"shotsAgainst": int(28 * defense[t] * gp)}]}
    if parts[0] == "standings":
        return {"standings": [{"teamAbbrev": {"default": t},
                               "gamesPlayed": sum(1 for d in calendrier for g in calendrier[d]
                                                  if t in g[1:] and etat.get(g[0]) == "OFF")} for t in EQUIPES]}
    if parts[0] == "gamecenter":
        return box.get(int(parts[1]), {"gameState": "FUT"})
    raise ValueError(path)


def main():
    tmp = Path(tempfile.mkdtemp())
    run.DIR_PRED = tmp / "data" / "predictions"
    run.DIR_CACHE = tmp / "data" / "cache"
    run.SORTIE = tmp / "docs" / "data.json"
    api.get_json = faux_get_json
    api._cache.clear()

    jours = sorted(calendrier)
    for d in jours:
        api._cache.clear()           # chaque "exécution" repart à neuf
        sys.argv = ["run.py", d]
        run.main()                   # prédit les matchs du soir
        if d != jours[-1]:
            jouer(d)                 # les matchs se jouent pendant la nuit

    out = json.loads(run.SORTIE.read_text())
    for v, s in out["stats"].items():
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()
                  if k in ("n", "base", "brier", "bss", "note")}, "top10", s.get("top10"))
    assert out["stats"]["v1"]["n"] > 300, "pas assez de prédictions gradées"
    assert out["joueurs"], "aucun joueur pour ce soir"
    # Copie pour prévisualiser la page avec des données réalistes
    dest = ICI / "data_exemple.json"
    shutil.copy(run.SORTIE, dest)
    print("OK ->", dest)


if __name__ == "__main__":
    main()
