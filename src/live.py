"""
Mise à jour LIVE pendant les matchs (roule aux 5 minutes via la GitHub Action).

Écrit docs/live.json : pour chaque match commencé ce soir, l'état (période,
temps), le score et les stats de chaque joueur (tirs, buts, passes). La page
s'en sert pour colorer les joueurs :
  vert  = objectif atteint (même si le match continue, c'est acquis)
  jaune = match en cours, pas encore atteint
  rouge = match fini, raté

POURQUOI un fichier séparé de data.json : il change aux 5 minutes et ne doit
jamais toucher aux prédictions (gelées à la mise au jeu). Petit et jetable.

POURQUOI on n'écrit que si quelque chose a changé : chaque écriture = un commit
= une republication de la page. Pas de changement, pas de bruit.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import nhl_api as api

SORTIE = Path(__file__).resolve().parent.parent / "docs" / "live.json"
AVANT = {"FUT", "PRE"}


def jour_soiree():
    """Date de la soirée en cours. Avant 6 h du matin, on est encore dans la
    soirée d'hier (un match de 22 h à Vegas finit après minuit chez nous)."""
    maintenant = datetime.now(ZoneInfo("America/Toronto"))
    if maintenant.hour < 6:
        maintenant -= timedelta(days=1)
    return maintenant.date().isoformat()


def etat_match(box):
    pd = box.get("periodDescriptor", {}) or {}
    clock = box.get("clock", {}) or {}
    pb = box.get("playerByGameStats", {})
    joueurs, buts = {}, {"awayTeam": 0, "homeTeam": 0}
    for cote in ("awayTeam", "homeTeam"):
        for grp in ("forwards", "defense"):
            for j in pb.get(cote, {}).get(grp, []):
                joueurs[str(j["playerId"])] = [j.get("sog", 0), j.get("goals", 0), j.get("assists", 0)]
                buts[cote] += j.get("goals", 0)
    return {
        "state": box.get("gameState"),
        "period": pd.get("number"), "periodType": pd.get("periodType"),
        "clock": clock.get("timeRemaining"), "intermission": clock.get("inIntermission", False),
        # Score officiel si présent; sinon somme des buts des joueurs (fallback)
        "hs": box.get("homeTeam", {}).get("score", buts["homeTeam"]),
        "as": box.get("awayTeam", {}).get("score", buts["awayTeam"]),
        "joueurs": joueurs,
    }


def main():
    jour = jour_soiree()
    sched = api.schedule(jour) or {}
    jeux = next((d["games"] for d in sched.get("gameWeek", []) if d["date"] == jour), [])
    commences = [g for g in jeux if g.get("gameState") not in AVANT]
    if not commences:
        print(f"{jour}: aucun match commencé")
        return

    ancien = {}
    try:
        ancien = json.loads(SORTIE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    games = {}
    for g in commences:
        box = api.boxscore(g["id"])
        if box:
            games[str(g["id"])] = etat_match(box)

    if ancien.get("date") == jour and ancien.get("games") == games:
        print(f"{jour}: rien de neuf")
        return
    SORTIE.write_text(json.dumps({
        "maj": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": jour, "games": games,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"{jour}: {len(games)} matchs mis à jour")


if __name__ == "__main__":
    main()
