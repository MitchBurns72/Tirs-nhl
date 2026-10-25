"""
Données supplémentaires pour la v2 (api.nhle.com/stats/rest, gratuit, sans clé).

Deux rapports par match, vérifiés le 2026-10-20 :
  - skater/realtime : totalShotAttempts (tirs cadrés + ratés + bloqués), missedShots...
    (+ hits, les mises en échec, pour l'indice de robustesse des équipes, 25 oct.)
  - skater/timeonice : ppTimeOnIce, shTimeOnIce, evTimeOnIce (en secondes), teamAbbrev

POURQUOI les tentatives : un tir cadré dépend en partie de la chance (le tir qui
frappe le poteau ne compte pas). Les tentatives sont ~2x plus nombreuses, donc
le taux d'un joueur se stabilise beaucoup plus vite, surtout en début de saison.

POURQUOI le temps en PP : un joueur du 1er avantage numérique tire beaucoup plus
par minute qu'à 5 contre 5. Et une équipe indisciplinée donne plus de PP à l'adversaire.

Tout est mis en cache : un match fini ne change plus, on le télécharge une fois.
"""

import json
from pathlib import Path

import nhl_api as api

STATS = "https://api.nhle.com/stats/rest/en"


def _lire(f, defaut):
    try:
        return json.loads(Path(f).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return defaut


def _ecrire(f, data):
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    Path(f).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _rapport(nom, cayenne, par_match, debut=0):
    """URL d'un rapport de stats (100 lignes max par page)."""
    from urllib.parse import quote
    return (f"{STATS}/skater/{nom}?isAggregate=false&isGame={'true' if par_match else 'false'}"
            f"&limit=100&start={debut}&cayenneExp={quote(cayenne)}")


def _toutes_pages(nom, cayenne, par_match):
    """Toutes les pages d'un rapport. Garde-fou : une ligne déjà vue (même joueur,
    même équipe, même match) n'est jamais comptée deux fois, et si une page revient
    identique à la précédente, on arrête (pagination ignorée = sinon totaux doublés)."""
    lignes, vus, debut, premiere = [], set(), 0, None
    while True:
        rep = api.get_json(_rapport(nom, cayenne, par_match, debut)) or {}
        page = rep.get("data", [])
        if not page or (premiere is not None and page[0] == premiere):
            return lignes
        premiere = page[0]
        for r in page:
            cle = (r.get("playerId"), r.get("teamAbbrev") or r.get("teamAbbrevs"), r.get("gameId"))
            if cle not in vus:
                vus.add(cle)
                lignes.append(r)
        debut += 100
        if len(page) < 100 or debut >= rep.get("total", 0):
            return lignes


def stats_d_un_match(gid):
    """{"j": {pid: [tentatives, pp_min, sh_min]}, "pp": {équipe: minutes de PP de l'équipe}}.
    Le temps de PP d'une équipe ≈ le plus gros temps de PP d'un de ses joueurs."""
    rt = _toutes_pages("realtime", f"gameId={gid}", True)
    ti = _toutes_pages("timeonice", f"gameId={gid}", True)
    if not rt and not ti:
        return None
    j, pp_eq, mises = {}, {}, {}
    for r in rt:
        j.setdefault(str(r["playerId"]), [None, 0.0, 0.0])[0] = r.get("totalShotAttempts")
        # Mises en échec données par l'équipe (robustesse). Champ « hits » du rapport realtime :
        # s'il est absent, l'équipe n'a juste pas d'indice (neutre).
        if r.get("hits") is not None and r.get("teamAbbrev"):
            mises[r["teamAbbrev"]] = mises.get(r["teamAbbrev"], 0) + r["hits"]
    for r in ti:
        e = j.setdefault(str(r["playerId"]), [None, 0.0, 0.0])
        e[1] = round((r.get("ppTimeOnIce") or 0) / 60, 2)
        e[2] = round((r.get("shTimeOnIce") or 0) / 60, 2)
        t = r.get("teamAbbrev")
        if t:
            pp_eq[t] = max(pp_eq.get(t, 0.0), e[1])
    return {"j": j, "pp": pp_eq, "h": mises}


def cache_stats_jeux(dossier, saison, ids, max_nouveaux=400):
    """Stats par match de la saison, en cache. Au plus `max_nouveaux` matchs
    téléchargés par exécution (le reste au prochain calcul)."""
    f = Path(dossier) / f"stats_jeux_{saison}.json"
    cache = _lire(f, {})
    # Les matchs en cache d'avant le 25 oct. n'ont pas les mises en échec ("h") : on les reprend
    manquants = [str(i) for i in ids if str(i) not in cache or "h" not in cache[str(i)]][:max_nouveaux]
    if manquants:
        chemins = {}
        for gid in manquants:
            chemins[gid] = (_rapport("realtime", f"gameId={gid}", True), _rapport("timeonice", f"gameId={gid}", True))
        api.fetch_many([u for paire in chemins.values() for u in paire])   # en parallèle, va dans le cache mémoire
        for gid in manquants:
            r = stats_d_un_match(gid)
            if r:
                cache[gid] = r
        _ecrire(f, cache)
    return cache


def cache_stats_saison(dossier, saison):
    """Totaux de la saison passée par joueur : {pid: [tentatives, pp_min, matchs]}.
    Un joueur échangé a une ligne par équipe : on additionne."""
    f = Path(dossier) / f"stats_saison_{saison}.json"
    cache = _lire(f, None)
    if cache is not None:
        return cache
    cay = f"seasonId={saison} and gameTypeId=2"
    out = {}
    for r in _toutes_pages("realtime", cay, False):
        e = out.setdefault(str(r["playerId"]), [0, 0.0, 0])
        e[0] += r.get("totalShotAttempts") or 0
        e[2] = max(e[2], r.get("gamesPlayed") or 0)
    for r in _toutes_pages("timeonice", cay, False):
        e = out.setdefault(str(r["playerId"]), [0, 0.0, 0])
        e[1] += round((r.get("ppTimeOnIce") or 0) / 60, 2)
        e[2] = max(e[2], r.get("gamesPlayed") or 0)
    if out:
        _ecrire(f, out)
    return out


def equipes_saison_passee(dossier, saison):
    """Par équipe, saison passée : [matchs, tirs pour, tirs contre, buts pour, buts contre].
    Tirs/buts pour = somme des joueurs ; tirs/buts contre = gardiens."""
    f = Path(dossier) / f"equipes2_{saison}.json"
    cache = _lire(f, {})
    st = api.standings_now() or {"standings": []}
    for s in st["standings"]:
        t = s["teamAbbrev"]["default"]
        if t in cache:
            continue
        cs = api.club_stats(t, saison)
        if not cs:
            cache[t] = None
            continue
        sk, go = cs.get("skaters", []), cs.get("goalies", [])
        cache[t] = [82, sum(x.get("shots", 0) for x in sk), sum(x.get("shotsAgainst", 0) for x in go),
                    sum(x.get("goals", 0) for x in sk), sum(x.get("goalsAgainst", 0) for x in go)] if sk else None
    _ecrire(f, cache)
    return cache
