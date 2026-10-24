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
from statistics import median as statistics_median
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
import auto  # noqa: E402

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
                        "playmaker": rng.uniform(0.5, 2.0),
                        # v2 : 1er/2e avantage numérique (4 avants + 2 déf. par équipe) et précision
                        "pp": rng.uniform(2.0, 3.5) if (i < 4 or i in (12, 13)) else rng.uniform(0.0, 0.6),
                        "ratio": rng.uniform(0.45, 0.65)}
K_PP_VRAI = 1.2                                   # une minute de PP vaut 2.2 minutes pour les tirs
indisc = {t: rng.uniform(0.7, 1.4) for t in EQUIPES}   # >1 : donne plus de PP à l'adversaire
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
stats_sim = {}   # gameId -> {pid: (tentatives, min PP, équipe)}


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
            ppm = max(0.0, j["pp"] * indisc[opp] * rng.uniform(0.6, 1.4))
            lam_sog = j["rate"] * (toi + K_PP_VRAI * ppm) * defense[opp]
            att = poisson(lam_sog / j["ratio"])
            s = sum(1 for _ in range(att) if rng.random() < j["ratio"])   # cadrés = tentatives × précision (bruit)
            lignes[p] = {"team": j["team"], "opp": opp, "toi": toi, "shots": s, "goals": 0, "assists": 0,
                         "att": att, "pp": round(ppm, 2)}
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
        stats_sim[gid] = {p: (l["att"], l["pp"], l["team"]) for p, l in lignes.items()}
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


def faux_stats(path):
    from urllib.parse import urlparse, parse_qs
    q = parse_qs(urlparse(path).query)
    nom = urlparse(path).path.rstrip("/").split("/")[-1]
    cay, debut = q["cayenneExp"][0], int(q.get("start", ["0"])[0])
    rows = []
    if cay.startswith("gameId="):
        for p, (att, ppm, team) in stats_sim.get(int(cay.split("=")[1]), {}).items():
            rows.append({"playerId": p, "teamAbbrev": team, "totalShotAttempts": att, "ppTimeOnIce": int(ppm * 60), "shTimeOnIce": 0})
    elif cay.startswith(f"seasonId={PREV}"):
        tot = {}
        for g, a, h in cal_prev:
            for p, (att, ppm, team) in stats_sim.get(g, {}).items():
                e = tot.setdefault(p, [0, 0.0, 0]); e[0] += att; e[1] += ppm; e[2] += 1
        for p, (att, ppm, n) in tot.items():
            rows.append({"playerId": p, "totalShotAttempts": att, "ppTimeOnIce": int(ppm * 60), "gamesPlayed": n})
    return {"data": rows[debut:debut + 100], "total": len(rows)} if nom in ("realtime", "timeonice") else {"data": [], "total": 0}


def faux_get_json(path, retries=4):
    if path.startswith("https://api.nhle.com/stats/rest/"):
        return faux_stats(path)
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
                           "positionCode": j["pos"], "sweaterNumber": p % 98 + 1, "headshot": None}
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
        sf = gf = 0
        for g in ids:
            cote_moi = "homeTeam" if bx[g]["homeTeam"]["abbrev"] == t else "awayTeam"
            st = bx[g]["playerByGameStats"][cote_moi]
            sf += sum(x["sog"] for grp in st.values() for x in grp)
            gf += sum(x["goals"] for grp in st.values() for x in grp)
        if prev:   # run.py divise par 82 : on ramène le total à l'échelle de 82 matchs
            k = 82 / max(len(ids), 1)
            sa, ga, sf, gf = int(sa * k), int(ga * k), int(sf * k), int(gf * k)
        return {"goalies": [{"shotsAgainst": sa, "goalsAgainst": ga}], "skaters": [{"shots": sf, "goals": gf}]}
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


def _bruit(p, rr):
    z = math.log(p / (1 - p)) + rr.gauss(0, 0.3)
    return 1 / (1 + math.exp(-z))


def _p_seuil_poisson(lam, k):
    pn, c = math.exp(-lam), math.exp(-lam)
    for n in range(1, k):
        pn *= lam / n; c += pn
    return 1 - c


def _match_vrai(a, h):
    lh, la = 3 * attaque[h] * defense[a] * 1.03, 3 * attaque[a] * defense[h] * 0.97
    ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(16)]
    pa = [math.exp(-la) * la ** i / math.factorial(i) for i in range(16)]
    win = sum(ph[i] * pa[j] for i in range(16) for j in range(16) if i > j)
    tie = sum(ph[i] * pa[i] for i in range(16))
    over = 1 - sum(ph[i] * pa[j] for i in range(16) for j in range(16) if i + j <= 6)
    return win + 0.52 * tie, over


def faux_odds(path, params):
    """Imite The Odds API (décimal) avec un marché qui connaît les vrais talents, bruité."""
    d = jour_courant[0]
    import zlib
    rr = random.Random(zlib.crc32(f"{d}|{path}".encode()))   # reproductible d'un test à l'autre
    evs = []
    for g, a, h in calendrier[d]:
        p, over = _match_vrai(a, h)
        p, over = _bruit(p, rr), _bruit(over, rr)
        evs.append({"id": f"ev{g}", "home_team": f"Ville {NOMS[h]}", "away_team": f"Ville {NOMS[a]}",
                    "bookmakers": [{"key": "bk", "markets": [
                        {"key": "h2h", "outcomes": [{"name": f"Ville {NOMS[h]}", "price": round(1 / (p * 1.045), 2)},
                                                    {"name": f"Ville {NOMS[a]}", "price": round(1 / ((1 - p) * 1.045), 2)}]},
                        {"key": "totals", "outcomes": [{"name": "Over", "point": 6.5, "price": round(1 / (over * 1.045), 2)},
                                                       {"name": "Under", "point": 6.5, "price": round(1 / ((1 - over) * 1.045), 2)}]}]}]})
    if path == "/odds":
        return evs
    gid = int(path.split("/")[2][2:])
    _, a, h = next(x for x in calendrier[d] if x[0] == gid)
    outs = []
    for p, j in joueurs.items():
        if j["team"] not in (a, h):
            continue
        opp = h if j["team"] == a else a
        lam = j["rate"] * (j["toi"] + K_PP_VRAI * j["pp"] * indisc[opp]) * defense[opp]
        for pt, k in ((0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4)):
            pk = min(max(_bruit(min(max(_p_seuil_poisson(lam, k), 0.01), 0.99), rr), 0.01), 0.97)
            outs.append({"name": "Over", "description": f"Joueur {str(p)[-3:]}", "point": pt, "price": round(1 / (pk * 1.08), 2)})
    return {"bookmakers": [{"key": "bk", "markets": [{"key": "player_shots_on_goal_alternate", "outcomes": outs}]}]}


jour_courant = [None]
BETS_TEMOIN = json.dumps([{"id": "temoin", "label": "Bet témoin", "cote": 1.9, "mise": 1, "cree": "2026-10-01T00:00:00Z"}])


def main():
    tmp = Path(tempfile.mkdtemp())
    run.DIR_PRED = tmp / "data" / "predictions"
    run.DIR_CACHE = tmp / "data" / "cache"
    run.ETAT_AUTO = tmp / "data" / "versions_auto.json"   # jamais dans le vrai repo
    run.SORTIE = tmp / "docs" / "data.json"
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    (tmp / "data" / "bets.json").write_text(BETS_TEMOIN)   # témoin : ne doit jamais bouger
    api.get_json = faux_get_json
    cotes._get = faux_odds
    def sgo_refuse(params):   # la clé de test n'est pas une clé SportsGameOdds -> repli sur The Odds API
        raise cotes.CleRefusee()
    cotes._sgo_get = sgo_refuse
    run.AVEC_COTES = True
    run.AVEC_ALIGNEMENTS = False   # pas d'internet dans le test
    import os
    os.environ["ODDS_API_KEY"] = "test"

    auto.JOURS_GENERATION = 999   # le test force la fin de génération lui-même (plus bas)
    jours = sorted(calendrier)
    for d in jours:
        api._cache.clear()
        jour_courant[0] = d
        sys.argv = ["run.py", d]
        run.main()
        if d != jours[-1]:
            jouer(d)

    # --- Ce soir : 2 matchs déjà commencés avant le premier calcul -> rétro ---
    dernier = jours[-1]
    (run.DIR_PRED / f"{dernier}.json").unlink()
    for gid, _, _ in calendrier[dernier][:2]:
        etat[gid] = "LIVE"
    api._cache.clear()
    sys.argv = ["run.py", dernier]
    run.main()
    f = json.loads((run.DIR_PRED / f"{dernier}.json").read_text())
    retro_ids = {e["gameId"] for e in f["entries"] if e.get("retro")}
    assert retro_ids == {g for g, _, _ in calendrier[dernier][:2]}, retro_ids
    assert sum(1 for g in f["games"] if g.get("retro")) == 2
    assert len({e["gameId"] for e in f["entries"]}) == 4, "les matchs ouverts doivent rester"
    print("rétro de ce soir OK")
    for gid, _, _ in calendrier[dernier][:2]:
        etat.pop(gid)

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
    print("global  ", {v: round(s["note"], 1) for v, s in st["global"].items() if s.get("note") is not None})
    assert st["joueurs"]["tirs"]["v1"]["n"] > 300
    assert st["matchs"]["victoire"]["v1"]["n"] > 40
    assert out["games"] and out["joueurs"]
    assert out["games"][0].get("cotes", {}).get("p_home_marche"), "cotes de match non associées"
    assert any(e.get("cotes") for e in out["joueurs"]), "cotes joueurs non associées"
    # --- Le modèle paramétrique avec V1 doit redonner v1 (sinon le tuner optimise autre chose) ---
    from modeles import V1, modele_joueur_param, modele_match_param
    pass  # auto importé en haut du fichier
    ecart_max = 0.0
    for fic in sorted(run.DIR_PRED.glob("*.json")):
        d = json.loads(fic.read_text())
        for e in d["entries"]:
            if e.get("f") and "tirs" in e["preds"].get("v1", {}):
                r = modele_joueur_param(e["f"], V1)
                for m in ("tirs", "buts", "passes"):
                    ecart_max = max(ecart_max, abs(r[m]["p"] - e["preds"]["v1"][m]["p"]))
        for g in d.get("games", []):
            if g.get("f"):
                ecart_max = max(ecart_max, abs(modele_match_param(g["f"], V1)["p_home"] - g["preds"]["v1"]["p_home"]))
    print(f"écart max modèle paramétrique vs v1 : {ecart_max:.6f}")
    assert ecart_max < 2e-3, "le modèle paramétrique ne reproduit pas v1"

    # --- Labo : la génération 2 (v2.1 à v2.5) se termine et la génération 3 naît ---
    auto.MIN_JOUEURS, auto.MIN_MATCHS, auto.MIN_VAL_JOUEURS, auto.MIN_VAL_MATCHS = 500, 30, 200, 10
    hist = [json.loads(f.read_text()) for f in sorted(run.DIR_PRED.glob("*.json"))]
    labo, nouvelle = auto.cycle(hist, run.ETAT_AUTO, list(run.VERSIONS), jours[-1], force=True)
    print("journal :", labo["journal"][0]["msg"])
    print("patterns :", [(p["marche"], p["segment"], round(p["predit"], 2), round(p["reel"], 2), round(p["z"], 1)) for p in labo["patterns"][:4]])
    gen2 = labo["generations"][0]
    print("classement gén. 2 :", [(c["id"], round((c["gain"] or 0) * 100, 2), round(c["z"] or 0, 1)) for c in gen2["classement"]])
    print("principale :", labo["principale"], "· promo :", labo.get("promo"))
    assert nouvelle and labo["generation"]["n"] == 3, "la génération 3 n'est pas née"
    # va.1 (scan complet) rejoint le concours, mais n'a aucune ligne notée avant le 24 octobre
    assert gen2["n"] == 2 and {c["id"] for c in gen2["classement"]} == {f"v2.{i}" for i in range(1, 6)} | {"va.1"}
    assert next(c for c in gen2["classement"] if c["id"] == "va.1")["gain"] is None, "va.1 notée sur ses soirs de scan"
    parent = gen2["gagnant"]
    enfants = [f"v3.{i}" for i in range(1, 6)]
    assert labo["generation"]["membres"] == [parent] + enfants + ["va.1"], "va.1 (trop jeune) doit passer à la suivante"
    perdants = [c["id"] for c in gen2["classement"] if c["id"] not in (parent, labo["principale"], "va.1")]
    assert set(perdants) <= set(labo.get("retirees", [])), "les perdants doivent arrêter de prédire"
    # Les enfants sont vraiment différents du parent (sinon on teste 5 fois la même chose)
    params = [json.dumps(v["params"], sort_keys=True) for v in labo["versions"] if v["generation"] == 3]
    assert len(set(params)) == 5, "enfants identiques"
    for v in labo["versions"]:
        print(f"  {v['id']} (écart {v.get('ecart')}) : {v['desc'][:120]}")
    # Ambition : les enfants doivent donner des % vraiment différents du parent
    ecarts = sorted(v["ecart"] for v in labo["versions"] if v["generation"] == 3)
    assert ecarts[-1] >= 0.015 and sum(e >= 0.004 for e in ecarts) >= 3, f"enfants trop pareils : {ecarts}"
    # Facteur de patinoire présent dans les features (et rétro-rempli sur les vieux soirs)
    vieux_f = json.loads((run.DIR_PRED / f"{jours[3]}.json").read_text())
    assert all("rk" in e["f"]["x"] for e in vieux_f["entries"] if (e.get("f") or {}).get("x")), "facteur de patinoire manquant"

    # Le lendemain : les enfants prédisent, les perdants non, et aucun enfant sur un vieux soir
    import modeles
    run.charger_auto(run.ETAT_AUTO)
    api._cache.clear()
    sys.argv = ["run.py", jours[-1]]
    run.main()
    d = json.loads((run.DIR_PRED / f"{jours[-1]}.json").read_text())
    ouverts = [e for e in d["entries"] if not e.get("retro")]
    for vid in enfants + [parent]:
        assert any(vid in e["preds"] for e in ouverts), f"{vid} ne prédit pas"
    for vid in perdants:
        assert vid not in modeles.VERSIONS, f"{vid} retirée mais encore chargée"
    vieux = json.loads((run.DIR_PRED / f"{jours[2]}.json").read_text())
    # Rejoués sur toute la saison (sans réseau)...
    assert all(vid in e["preds"] for e in vieux["entries"] if (e.get("f") or {}).get("x") for vid in enfants), "enfants pas rejoués"
    # ...mais le Tuner / Calibreur ne sont pas notés sur leurs soirs d'entraînement
    vs = {v["id"]: v for v in labo["versions"]}
    ajustes = [vid for vid in enfants if vs[vid]["desc"].startswith("Tuner : raffine")
               or (vs[vid]["desc"].startswith("Calibreur") and "recalibr" in vs[vid]["desc"])]
    assert all(vs[vid]["juge_depuis"] for vid in ajustes), "juge_depuis manquant"
    assert all(vs[vid]["juge_depuis"] is None for vid in enfants if vid not in ajustes)
    st_s = json.loads(run.SORTIE.read_text())["stats"]["joueurs"]["tirs"]
    for vid in ajustes:
        assert st_s[vid]["n"] < st_s[parent]["n"], f"{vid} noté sur ses soirs d'entraînement"
    print("rétro des enfants OK · notés sur :", {vid: st_s[vid]["n"] for vid in enfants}, "vs parent", st_s[parent]["n"])
    # Leur historique reste : les perdants ont toujours leurs prédictions des vieux soirs
    assert any(perdants[0] in e["preds"] for e in vieux["entries"]), "historique des perdants effacé"
    lab = json.loads(run.SORTIE.read_text())["labo"]
    assert lab["generation"]["n"] == 3 and lab["attente"], "la génération 3 doit attendre ses soirs en direct"
    assert lab["attente"]["soirs_direct"] is not None and lab["attente"]["soirs_direct"] < auto.GEN_MIN_SOIRS_DIRECT
    print(f"génération 3 active ({', '.join(enfants)} vs {parent}), attend {auto.GEN_MIN_SOIRS_DIRECT} soirs en direct OK")

    # --- Migration : un état de l'ancien moteur (sans "moteur") doit relancer la génération ---
    e = json.loads(run.ETAT_AUTO.read_text())
    e.pop("moteur", None)
    run.ETAT_AUTO.write_text(json.dumps(e))
    hist = [json.loads(f.read_text()) for f in sorted(run.DIR_PRED.glob("*.json"))]
    labo, nouvelle = auto.cycle(hist, run.ETAT_AUTO, list(run.VERSIONS), jours[-1])
    print("migration :", labo["journal"][0]["msg"][:200])
    assert nouvelle and labo["generation"]["n"] == 4 and labo["generation"]["membres"][0] == parent
    assert labo["generations"][0].get("interrompue") and labo["moteur"] == 3
    assert all(v["retiree"] for v in labo["versions"] if v["generation"] == 3), "anciennes v3.x pas retirées"
    run.charger_auto(run.ETAT_AUTO)
    assert "v4.1" in modeles.VERSIONS and "v3.1" not in modeles.VERSIONS
    labo2, nouvelle2 = auto.cycle(hist, run.ETAT_AUTO, list(run.VERSIONS), jours[-1])
    assert not nouvelle2 and labo2["generation"]["n"] == 4, "la migration ne doit rouler qu'une fois"
    print("migration vers le nouveau moteur OK")
    # Rythme hebdo : 7 jours plus tard, mais sans 3 soirs en direct -> on attend encore
    auto.JOURS_GENERATION = 7
    from datetime import date as _d, timedelta as _td
    plus7 = (_d.fromisoformat(jours[-1]) + _td(days=7)).isoformat()
    labo3, n3 = auto.cycle(hist, run.ETAT_AUTO, list(run.VERSIONS), plus7)
    assert not n3 and labo3["attente"]["fin"] == plus7 and labo3["attente"]["soirs_direct"] == 0
    print("rythme hebdo OK (attend ses soirs en direct)")

    # --- Seuils de tirs 1+ à 4+ ---
    st = json.loads(run.SORTIE.read_text())
    for e in st["joueurs"]:
        for v, pv in e["preds"].items():
            ps = [pv[m]["p"] for m in ("tirs1", "tirs", "tirs3", "tirs4")]
            assert all(a >= b for a, b in zip(ps, ps[1:])), f"seuils pas décroissants {ps}"
    assert st["stats"]["joueurs"]["tirs4"]["v1"]["n"] > 300, "tirs 4+ pas gradés"
    assert any(e.get("cotes", {}) and "tirs3" in e["cotes"] for e in st["joueurs"]), "cotes 3+ pas lues"
    print("tirs 1+/3+/4+ :", {m: round(st["stats"]["joueurs"][m]["v1"]["note"], 1) for m in ("tirs1", "tirs", "tirs3", "tirs4")})
    # Une vieille prédiction (format du 17 octobre) reçoit ses seuils à partir de son λ
    vieux = {"date": "2026-10-01", "entries": [{"pid": 1, "gameId": 1, "preds": {"v0": {"p": 0.6, "lam": 2.0},
             "v1": {"p": 0.7, "lam": 2.5, "detail": {}}}, "result": 3, "played": True}], "games": []}
    (run.DIR_PRED / "2026-10-01.json").write_text(json.dumps(vieux))
    run.completer_seuils()
    pv = json.loads((run.DIR_PRED / "2026-10-01.json").read_text())["entries"][0]["preds"]
    from modeles import p_seuil
    assert abs(pv["v0"]["tirs3"]["p"] - p_seuil(2.0, 3)) < 1e-4 and abs(pv["v1"]["tirs4"]["p"] - p_seuil(2.5, 4, 15.0)) < 1e-4
    assert abs(p_seuil(2.0, 2) - (1 - 3 * 2.718281828 ** -2)) < 1e-6   # Poisson : 1 - e^-λ(1+λ)
    (run.DIR_PRED / "2026-10-01.json").unlink()
    print("complétion des vieilles prédictions OK")

    # --- Archives de résultats pour les bets ---
    arch = run.SORTIE.parent / "live" / f"{jours[-2]}.json"
    a = json.loads(arch.read_text())
    assert a["complet"] and len(a["games"]) == 4, "archive du soir précédent incomplète"
    un = next(iter(a["games"].values()))
    assert un["joueurs"] and len(next(iter(un["joueurs"].values()))) == 3
    print("archives des soirs OK")
    shutil.copy(arch, ICI / "archive_exemple.json")

    # --- Lecteur SportsGameOdds sur une réponse imitée des docs ---
    ev = {"teams": {"home": {"names": {"long": "Minnesota Wild"}}, "away": {"names": {"long": "Columbus Blue Jackets"}}},
          "players": {"BLAKE_COLEMAN_1_NHL": {"name": "Blake Coleman"}},
          "odds": {
            "points-home-game-ml-home": {"statID": "points", "statEntityID": "home", "periodID": "game", "betTypeID": "ml", "sideID": "home",
                                         "byBookmaker": {"draftkings": {"odds": "-180"}, "fanduel": {"odds": "-170"}}},
            "points-away-game-ml-away": {"statID": "points", "statEntityID": "away", "periodID": "game", "betTypeID": "ml", "sideID": "away",
                                         "byBookmaker": {"draftkings": {"odds": "+150"}, "fanduel": {"odds": "+145"}}},
            "points-all-reg-ml3way-home": {"statID": "points", "statEntityID": "all", "periodID": "reg", "betTypeID": "ml3way", "sideID": "home",
                                           "byBookmaker": {"draftkings": {"odds": "-120"}}},
            "points-all-game-ou-over": {"statID": "points", "statEntityID": "all", "periodID": "game", "betTypeID": "ou", "sideID": "over",
                                        "byBookmaker": {"draftkings": {"odds": "-140", "overUnder": "5.5",
                                                                       "altLines": [{"odds": "+110", "overUnder": "6.5"}]}}},
            "shots_onGoal-BLAKE_COLEMAN_1_NHL-game-ou-over": {"statID": "shots_onGoal", "statEntityID": "BLAKE_COLEMAN_1_NHL", "periodID": "game",
                "betTypeID": "ou", "sideID": "over", "bookOverUnder": "2.5",
                "byBookmaker": {"draftkings": {"odds": "+170", "overUnder": "2.5",
                                               "altLines": [{"odds": "-185", "overUnder": "1.5"}, {"odds": "+425", "overUnder": "3.5"}]}}}}}
    cm, cj = cotes.analyser_sgo([ev], [{"id": 9, "home": "MIN", "away": "CBJ"}], {"MIN": "Wild", "CBJ": "Blue Jackets"})
    assert cm[9]["home"] == round(statistics_median([1 + 100 / 180, 1 + 100 / 170]), 3), cm[9]
    assert cm[9]["away"] == 2.475 and cm[9]["h3"] == round(1 + 100 / 120, 3)
    assert cm[9]["totaux"]["5.5"]["over"] == round(1 + 100 / 140, 3) and cm[9]["totaux"]["6.5"]["over"] == 2.1
    jc = cj[(9, "blake coleman")]
    assert jc["tirs3"]["prix"] == 2.7 and jc["tirs"]["prix"] == round(1 + 100 / 185, 3) and jc["tirs4"]["prix"] == 5.25
    print("lecteur SportsGameOdds OK :", {k: v["prix"] for k, v in jc.items()})

    # --- Diagnostic : le marché seul vs v1 vs v2 (avant / après mélange), sur les tirs 2+ ---
    V2 = "v2.1"   # la référence v2 (mêmes paramètres que l'ancienne v2)
    diag = {"mkt": [], "v1": [], "v2": [], "v2_avant": []}
    n_cotes = n_tot = 0
    for fic in sorted(run.DIR_PRED.glob("*.json")):
        for e in json.loads(fic.read_text())["entries"]:
            if not e.get("played") or not isinstance(e.get("result"), dict):
                continue
            n_tot += 1
            c = (e.get("cotes") or {}).get("tirs")
            if not c or V2 not in e["preds"]:
                continue
            n_cotes += 1
            y = 1 if e["result"]["tirs"] >= 2 else 0
            pm = min(1 / c["prix"] / 1.08, 0.995)
            diag["mkt"].append((pm - y) ** 2)
            diag["v1"].append((e["preds"]["v1"]["tirs"]["p"] - y) ** 2)
            diag["v2"].append((e["preds"][V2]["tirs"]["p"] - y) ** 2)
            diag["v2_avant"].append((e["preds"][V2]["tirs"].get("p0", e["preds"][V2]["tirs"]["p"]) - y) ** 2)
            j = joueurs[e["pid"]]
            lam_vrai = j["rate"] * (j["toi"] + K_PP_VRAI * j["pp"] * indisc[e["opp"]]) * defense[e["opp"]]
            diag.setdefault("vrai", []).append((_p_seuil_poisson(lam_vrai, 2) - y) ** 2)
    # Données v2 bien branchées? (part des joueurs avec features x, PP projeté vs vrai rôle)
    avec_x = pp_ok = 0; ecarts = []; pens = []
    for fic in sorted(run.DIR_PRED.glob("*.json"))[-5:]:
        for e in json.loads(fic.read_text())["entries"]:
            det = e["preds"].get(V2, {}).get("tirs", {}).get("detail")
            if e.get("f", {}).get("x") and det:
                avec_x += 1
                vrai_pp = joueurs[e["pid"]]["pp"] * indisc[e["opp"]]
                ecarts.append(abs(det["pp"] - vrai_pp)); pens.append((det["f_pen"], indisc[e["opp"]]))
    import statistics as stt
    corr = stt.correlation([a for a, b in pens], [b for a, b in pens]) if len(pens) > 2 else None
    print(f"v2 branchée : {avec_x} joueurs avec features x · erreur PP moyenne {stt.mean(ecarts):.2f} min · "
          f"corrélation indiscipline détectée/vraie {corr:.2f}")
    print(f"diagnostic tirs 2+ ({n_cotes}/{n_tot} avec cotes) :", {k: round(sum(v) / len(v), 4) for k, v in diag.items() if v})

    # --- v2.1 à v2.5 : rejouées sur l'historique, v2.1 meilleure que v1 sur les tirs ---
    st2 = json.loads(run.SORTIE.read_text())["stats"]
    for m in ("tirs", "tirs3", "buts", "passes"):
        print(f"v2.1 vs v1 {m:7}", round(st2["joueurs"][m][V2]["note"], 1), "vs", round(st2["joueurs"][m]["v1"]["note"], 1),
              f"(n={st2['joueurs'][m][V2]['n']})")
    print("v2.1 vs v1 victoire", round(st2["matchs"]["victoire"][V2]["note"], 1), "vs", round(st2["matchs"]["victoire"]["v1"]["note"], 1))
    print("note globale", {v: round(s["note"], 1) for v, s in st2["global"].items() if s.get("note") is not None})
    assert st2["joueurs"]["tirs"][V2]["n"] >= st2["joueurs"]["tirs"]["v1"]["n"] * 0.95, "v2.1 pas rejouée sur l'historique"
    assert st2["joueurs"]["tirs"][V2]["note"] > st2["joueurs"]["tirs"]["v1"]["note"], "v2.1 ne bat pas v1 sur les tirs"
    # Rejouées sans réseau : toutes les lignes avec features ont les 5 versions de la génération 2
    for fic in sorted(run.DIR_PRED.glob("*.json"))[:-1]:
        for e in json.loads(fic.read_text())["entries"]:
            if e.get("f", {}).get("x"):
                assert all(f"v2.{i}" in e["preds"] for i in range(1, 6)), f"{fic.name} : génération 2 incomplète"
    # Champions par catégorie + version combo
    sortie0 = json.loads(run.SORTIE.read_text())
    P = sortie0["principales"]
    print("champions :", P, "| promos :", {c: (p["candidat"], round((p["gain"] or 0) * 100, 2), p["z"] and round(p["z"], 1))
                                          for c, p in (sortie0["labo"].get("promos") or {}).items()})
    assert set(P) == {"tirs", "buts", "passes", "victoire", "totaux"} and sortie0["principale"] == "combo"
    e0 = next(e for e in sortie0["joueurs"] if "combo" in e["preds"] and P["passes"] in e["preds"])
    assert e0["preds"]["combo"]["passes"]["p"] == e0["preds"][P["passes"]]["passes"]["p"]
    assert e0["preds"]["combo"]["tirs3"]["src"] == P["tirs"]
    g0 = next(g for g in sortie0["games"] if "combo" in g["preds"])
    assert g0["preds"]["combo"]["p_home"] == g0["preds"][P["victoire"]]["p_home"]
    assert g0["preds"]["combo"]["p_over"] == g0["preds"][P["totaux"]]["p_over"]
    assert all(len(sortie0["labo"]["classement_cat"][c]) >= 3 for c in P)
    print("combo OK")
    # Fiche joueur : numéro et stats de saison avec projection sur les matchs restants
    sj = e0["saison"]
    assert e0["no"] and sj["pj"] > 0 and sj["proj"]["tirs"] >= sj["tirs"] and sj["max_pj"] == sj["pj"] + sj["restants"]
    print("stats de saison OK :", e0["name"], sj)
    # Blessures : lecteur ESPN sur une réponse imitée, puis annotation
    import alignements
    faux_espn = {"injuries": [{"displayName": "Montreal Canadiens", "injuries": [
        {"status": "Out", "athlete": {"displayName": e0["name"], "team": {"abbreviation": "MTL"}},
         "details": {"type": "Lower Body", "returnDate": "2026-11-01"}}]}]}
    alignements._texte = lambda url, timeout=25: json.dumps(faux_espn)
    bl = alignements.blessures()
    copie = [dict(e0)]
    assert alignements.annoter(copie, bl) == 1 and copie[0]["blesse"]["statut"] == "Out" and alignements.absent(bl[next(iter(bl))])
    print("blessures ESPN OK")
    # Lecteurs DailyFaceoff sur la structure réelle (vue dans la sonde du 24 octobre)
    comb = {"updatedAt": "2026-10-23T23:41:46Z", "sourceName": "Eric Engels", "players": [
        {"name": "Cole Caufield", "jerseyNumber": 13, "positionIdentifier": "lw", "groupIdentifier": "f1", "fantasydataFaceUrl": "u1"},
        {"name": "Cole Caufield", "jerseyNumber": 13, "positionIdentifier": "lw", "groupIdentifier": "pp1", "fantasydataFaceUrl": "u1"},
        {"name": "Lane Hutson", "jerseyNumber": 48, "positionIdentifier": "ld", "groupIdentifier": "d1"},
        {"name": "Ivan Demidov", "positionIdentifier": "rw", "groupIdentifier": "ir", "injuryStatus": "ir"}]}
    tr = alignements.lire_trios(comb)
    assert tr["groupes"] == {"f1": ["Cole Caufield"], "d1": ["Lane Hutson"], "pp1": ["Cole Caufield"], "ir": ["Ivan Demidov"]}
    assert tr["joueurs"]["Ivan Demidov"]["inj"] == "ir" and tr["joueurs"]["Cole Caufield"]["no"] == 13
    gd = alignements.lire_gardiens({"data": [{"homeTeamSlug": "chicago-blackhawks", "homeGoalieName": "Spencer Knight",
        "homeNewsStrengthName": "Confirmed", "homeGoalieSavePercentage": "0.888", "awayTeamSlug": "montreal-canadiens",
        "awayGoalieName": "Jacob Fowler", "awayNewsStrengthName": "Likely"}]})
    assert gd["CHI"]["nom"] == "Spencer Knight" and gd["MTL"]["statut"] == "Likely"
    print("lecteurs DailyFaceoff OK")
    # « Pourquoi ce % » : la v2 explique aussi but et passe, et le mélange avec le marché
    sortie = json.loads(run.SORTIE.read_text())
    pv = next(e["preds"][parent] for e in sortie["joueurs"] if parent in e["preds"] and e.get("cotes", {}).get("tirs"))
    assert {"r_sog", "r_att", "ratio", "w_att"} <= set(pv["tirs"]["detail"]), pv["tirs"]["detail"]
    assert "pp" in pv["buts"]["detail"] and "pp" in pv["passes"]["detail"]
    assert pv["tirs"]["p0"] is not None and pv["tirs"]["mkt"] is not None and pv["tirs"]["w"] is not None
    print("détail « Pourquoi ce % » v2 OK · taille data.json", round(run.SORTIE.stat().st_size / 1e6, 2), "Mo")
    # Les bets : aucun fichier Python n'y touche, et le fichier reste identique
    for fpy in (ICI.parent / "src").glob("*.py"):
        assert "bets.json" not in fpy.read_text(encoding="utf-8"), f"{fpy.name} touche aux bets"
    assert (tmp / "data" / "bets.json").read_text() == BETS_TEMOIN, "data/bets.json a été modifié"
    print("bets intacts OK")

    dest = ICI / "data_exemple.json"
    shutil.copy(run.SORTIE, dest)
    print("OK ->", dest)


if __name__ == "__main__":
    main()
