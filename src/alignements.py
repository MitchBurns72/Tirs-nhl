"""
Infos du soir : blessures (ESPN), et SONDES pour les gardiens partants et les trios /
unités d'avantage numérique (DailyFaceoff).

POURQUOI des sondes d'abord : DailyFaceoff est un site Next.js. Ses données sont dans un
bloc JSON (__NEXT_DATA__) dont on ne connaît pas encore les noms de champs. Plutôt que de
deviner (et de lire n'importe quoi en silence), on sauve le JSON brut une fois par jour
dans data/sondes/. Le lecteur sera écrit sur la VRAIE structure à l'étape suivante.

Tout ici est optionnel : si un site plante ou change, le calcul continue sans ces infos.
"""

import json
import re
import urllib.request
from datetime import date
from pathlib import Path

from cotes import norm

ESPN_BLESSURES = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
DFO_GARDIENS = "https://www.dailyfaceoff.com/starting-goalies"
DFO_TRIOS = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
# Statuts ESPN qui veulent dire « ne jouera pas ce soir »
ABSENTS = {"out", "injured reserve", "ir", "suspension", "suspended", "long term injured reserve", "ltir"}


def _texte(url, timeout=25):
    # Navigateur ordinaire : certains sites refusent les robots sans User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Linux; Android 13) tirs-nhl/1.0",
                                               "Accept": "text/html,application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def blessures():
    """{nom normalisé: {"statut", "detail", "retour", "equipe"}} d'après ESPN. {} si ça plante."""
    try:
        data = json.loads(_texte(ESPN_BLESSURES))
    except Exception as ex:  # noqa: BLE001
        print(f"  blessures ESPN indisponibles : {str(ex)[:100]}")
        return {}
    out = {}
    for groupe in data.get("injuries", []) or []:
        for b in groupe.get("injuries", []) or []:
            ath = b.get("athlete") or {}
            nom = norm(ath.get("displayName"))
            if not nom:
                continue
            det = b.get("details") or {}
            out[nom] = {"statut": b.get("status") or (b.get("type") or {}).get("description") or "?",
                        "detail": " ".join(x for x in (det.get("type"), det.get("detail")) if x) or b.get("shortComment", "")[:80],
                        "retour": det.get("returnDate"),
                        "equipe": ((ath.get("team") or {}).get("abbreviation") or groupe.get("displayName"))}
    return out


def absent(info):
    return bool(info) and str(info.get("statut", "")).lower() in ABSENTS


def annoter(entries, bless):
    """Ajoute "blesse" aux joueurs listés par ESPN. On ne les retire pas : si ESPN se
    trompe, on ne perd rien, et un joueur qui ne joue pas est annulé au grading."""
    n = 0
    for e in entries:
        info = bless.get(norm(e["name"]))
        if info:
            e["blesse"] = {k: info[k] for k in ("statut", "detail", "retour")}
            n += 1
    return n


def _next_data(html):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1)) if m else None


def sonder(dossier, jour):
    """Une fois par jour : sauve le JSON brut de DailyFaceoff (gardiens + trios d'une
    équipe) et d'ESPN, pour écrire les lecteurs sur la vraie structure."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    marque = dossier / "date.txt"
    if marque.exists() and marque.read_text().strip() == jour:
        return
    rapport = {"jour": jour}
    for nom, url in (("dfo_gardiens", DFO_GARDIENS),
                     ("dfo_trios_mtl", DFO_TRIOS.format(slug="montreal-canadiens"))):
        try:
            html = _texte(url)
            nd = _next_data(html)
            if nd is not None:
                (dossier / f"{nom}.json").write_text(json.dumps(nd, ensure_ascii=False)[:3_000_000], encoding="utf-8")
                rapport[nom] = "ok (__NEXT_DATA__)"
            else:
                (dossier / f"{nom}.html").write_text(html[:1_500_000], encoding="utf-8")
                rapport[nom] = "ok (html, pas de __NEXT_DATA__)"
        except Exception as ex:  # noqa: BLE001
            rapport[nom] = f"erreur : {str(ex)[:150]}"
    try:
        (dossier / "espn_blessures.json").write_text(_texte(ESPN_BLESSURES)[:2_000_000], encoding="utf-8")
        rapport["espn"] = "ok"
    except Exception as ex:  # noqa: BLE001
        rapport["espn"] = f"erreur : {str(ex)[:150]}"
    (dossier / "rapport.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=1), encoding="utf-8")
    marque.write_text(jour)
    print("  sondes :", rapport)


if __name__ == "__main__":
    sonder(Path(__file__).resolve().parent.parent / "data" / "sondes", date.today().isoformat())
