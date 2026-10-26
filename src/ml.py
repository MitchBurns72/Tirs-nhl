"""
Machine learning : un modèle par catégorie, entraîné sur TOUTES les données qu'on ramasse.

L'idée de Mitch (25 oct.) : au lieu de régler à la main une recette avec quelques ratios,
on donne au modèle tout ce qu'on peut récolter (forme, temps de glace, PP, tentatives,
adversaire, gardiens, robustesse, patinoire, repos, voyage, météo...) et il trouve lui-même
les combinaisons qui prédisent. Ce qui ne sert à rien (la météo, probablement) finit à 0.

Modèle : gradient boosting (HistGradientBoostingClassifier de scikit-learn). Des centaines
de petits arbres de décision, chacun corrige les erreurs des précédents. Il gère les
données manquantes tout seul (ex. les tentatives de tirs avant qu'on les ait).

Données : la saison en cours + les 3 saisons passées (API NHL, gratuite). Récoltées une
fois, en cache dans data/hist/<saison>/. La première récolte prend quelques calculs.

Honnêteté (même juge que le reste) :
  - Chaque prédiction d'un soir passé vient d'un modèle entraîné SEULEMENT sur les matchs
    d'avant la semaine de ce soir-là (« walk-forward »). Le Suivi est donc honnête partout.
  - Le modèle remplace la recette en service d'une catégorie seulement s'il fait mieux sur
    les soirs récents qu'il n'a jamais vus (décision hebdo, voir auto.py).
"""

import json
import math
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path

import donnees_v2
import nhl_api as api

RACINE = Path(__file__).resolve().parent.parent
DIR_HIST = RACINE / "data" / "hist"
SAISONS_PASSEES = ["20252026", "20242025", "20232024"]
BUDGET_RECOLTE_S = 480      # secondes de récolte max par calcul (le reste au prochain)
MIN_LIGNES = 3000           # lignes joueur-match minimum avant d'entraîner (les tests le baissent)
MIN_MATCHS = 300

# Ville de chaque aréna : (latitude, longitude, fuseau UTC en hiver) -> voyage, décalage, météo
COORD = {"ANA": (33.81, -117.88, -8), "BOS": (42.37, -71.06, -5), "BUF": (42.88, -78.88, -5),
         "CGY": (51.04, -114.05, -7), "CAR": (35.80, -78.72, -5), "CHI": (41.88, -87.67, -6),
         "COL": (39.75, -105.01, -7), "CBJ": (39.97, -83.01, -5), "DAL": (32.79, -96.81, -6),
         "DET": (42.34, -83.06, -5), "EDM": (53.55, -113.50, -7), "FLA": (26.16, -80.33, -5),
         "LAK": (34.04, -118.27, -8), "MIN": (44.94, -93.10, -6), "MTL": (45.50, -73.57, -5),
         "NSH": (36.16, -86.78, -6), "NJD": (40.73, -74.17, -5), "NYI": (40.72, -73.59, -5),
         "NYR": (40.75, -73.99, -5), "OTT": (45.30, -75.93, -5), "PHI": (39.90, -75.17, -5),
         "PIT": (40.44, -79.99, -5), "SJS": (37.33, -121.90, -8), "SEA": (47.62, -122.35, -8),
         "STL": (38.63, -90.20, -6), "TBL": (27.94, -82.45, -5), "TOR": (43.64, -79.38, -5),
         "UTA": (40.77, -111.90, -7), "ARI": (33.53, -112.26, -7), "VAN": (49.28, -123.11, -8),
         "VGK": (36.10, -115.18, -8), "WSH": (38.90, -77.02, -5), "WPG": (49.89, -97.14, -6)}

# Catégories joueur : (stat, seuil). Les matchs : victoire du local, totaux.
CATS_J = {"tirs1": ("tirs", 1), "tirs2": ("tirs", 2), "tirs3": ("tirs", 3), "tirs4": ("tirs", 4),
          "but": ("buts", 1), "but2": ("buts", 2), "but3": ("buts", 3),
          "passe": ("passes", 1), "passe2": ("passes", 2), "pts1": ("pts", 1), "pts2": ("pts", 2)}
MARCHE_DE = {"tirs1": "tirs1", "tirs2": "tirs", "tirs3": "tirs3", "tirs4": "tirs4", "but": "buts",
             "but2": "buts2", "but3": "buts3", "passe": "passes", "passe2": "passes2", "pts1": "pts1", "pts2": "pts2"}
LAM_DE = {"tirs": "tirs", "buts": "buts", "passes": "passes", "pts": "pts"}


def _sklearn():
    """scikit-learn n'est pas installé dans l'Action de base : on l'installe la 1re fois."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    return HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _lire(f, defaut):
    try:
        return json.loads(Path(f).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return defaut


def _ecrire(f, data):
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    Path(f).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def _toi(s):
    try:
        m, sec = (s or "0:00").split(":")
        return int(m) + int(sec) / 60
    except ValueError:
        return 0.0


# =================================================================================
# 1. Récolte
# =================================================================================

def _calendrier_saison(saison):
    """Tous les matchs de saison régulière finis : {gid: [date, local, visiteur, buts loc, buts vis]}."""
    out = {}
    for t in COORD:
        sch = api.club_schedule(t, saison) or {}
        for g in sch.get("games", []):
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL") or not g.get("id"):
                continue
            out[str(g["id"])] = [g["gameDate"], g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"],
                                 g["homeTeam"].get("score"), g["awayTeam"].get("score")]
    return out


def _compacter_box(b):
    """Feuille de match -> lignes joueur compactes + gardien partant de chaque équipe.
    Joueur : [pid, équipe, déf(0/1), toi, tirs, buts, passes, mises en échec, pim, bloqués, buts PP]."""
    pb = b.get("playerByGameStats") or {}
    lignes, gardiens = [], {}
    for cote in ("homeTeam", "awayTeam"):
        t = (b.get(cote) or {}).get("abbrev")
        for grp in ("forwards", "defense"):
            for j in (pb.get(cote) or {}).get(grp, []):
                lignes.append([j.get("playerId"), t, 1 if grp == "defense" else 0, round(_toi(j.get("toi")), 2),
                               j.get("sog", 0), j.get("goals", 0), j.get("assists", 0), j.get("hits"),
                               j.get("pim"), j.get("blockedShots"), j.get("powerPlayGoals")])
        for gd in (pb.get(cote) or {}).get("goalies", []):
            if gd.get("starter") or (t not in gardiens and _toi(gd.get("toi")) > 30):
                gardiens[t] = gd.get("playerId")
    return {"j": lignes, "g": gardiens}


def _meteo(saison, cal):
    """Température moyenne du jour dans la ville du match (Open-Meteo, gratuit, sans clé).
    Oui, c'est un sport d'intérieur : c'est le modèle qui dira que ça ne sert à rien."""
    out = {}
    par_ville = defaultdict(set)
    for gid, (d, h, *_r) in cal.items():
        par_ville[h].add(d)
    for t, dates in par_ville.items():
        if t not in COORD:
            continue
        lat, lon, _ = COORD[t]
        debut, fin = min(dates), max(dates)
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={debut}&end_date={fin}&daily=temperature_2m_mean&timezone=auto")
        try:
            r = api.get_json(url) or {}
            jours = r.get("daily", {})
            out[t] = dict(zip(jours.get("time", []), jours.get("temperature_2m_mean", [])))
        except Exception:  # noqa: BLE001
            continue
    return out


def _meteo_courante(cal, jour):
    """Saison en cours : archive (5 jours de retard) + prévisions pour les derniers jours et ce soir."""
    out = _meteo("courante", {g: v for g, v in cal.items() if v[0] < (date.fromisoformat(jour) - timedelta(days=6)).isoformat()})
    for t, (lat, lon, _) in COORD.items():
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&daily=temperature_2m_mean&past_days=7&forecast_days=2&timezone=auto")
        try:
            r = api.get_json(url) or {}
            jours = r.get("daily", {})
            out.setdefault(t, {}).update(dict(zip(jours.get("time", []), jours.get("temperature_2m_mean", []))))
        except Exception:  # noqa: BLE001
            continue
    return out


def recolter(saison_courante, jour, feuilles):
    """Récolte incrémentale, dans le budget de temps. Retourne la liste des saisons prêtes."""
    t0 = time.time()
    pretes = []
    # Saison en cours : calendrier = nos feuilles de match ; feuilles par joueur à jour chaque calcul
    for saison in [str(saison_courante)] + SAISONS_PASSEES:
        d = DIR_HIST / saison
        etat = _lire(d / "etat.json", {})
        if saison == str(saison_courante):
            cal = {g: [fb["date"], fb["home"], fb["away"], fb["hg"], fb["ag"]] for g, fb in feuilles.items() if fb["date"] < jour}
            _ecrire(d / "calendrier_courant.json", cal)
            if etat.get("meteo_jour") != jour:
                _ecrire(d / "meteo.json", _meteo_courante(cal, jour))
                etat["meteo_jour"] = jour
        elif etat.get("calendrier"):
            cal = _lire(d / "calendrier.json", {})
        else:
            if time.time() - t0 > BUDGET_RECOLTE_S:
                break
            cal = _calendrier_saison(saison)
            if not cal:
                continue
            _ecrire(d / "calendrier.json", cal)
            etat["calendrier"] = True
        box = _lire(d / "box.json", {})
        manquants = [g for g in cal if g not in box]
        while manquants and time.time() - t0 < BUDGET_RECOLTE_S:
            lot, manquants = manquants[:300], manquants[300:]
            api.fetch_many([f"/gamecenter/{g}/boxscore" for g in lot])
            for g in lot:
                b = api.boxscore(g)
                if b and b.get("playerByGameStats"):
                    box[g] = _compacter_box(b)
            _ecrire(d / "box.json", box)
        if manquants:
            _ecrire(d / "etat.json", etat)
            break
        # Tentatives + temps en PP par match : la saison en cours a déjà son cache (v2)
        if saison != str(saison_courante) and not etat.get("rt"):
            if time.time() - t0 > BUDGET_RECOLTE_S:
                _ecrire(d / "etat.json", etat)
                break
            rt = {}
            cay = f"seasonId={saison} and gameTypeId=2"
            for r in donnees_v2._toutes_pages("realtime", cay, True):
                rt.setdefault(str(r.get("gameId")), {}).setdefault(str(r["playerId"]), [None, 0.0, 0.0])[0] = r.get("totalShotAttempts")
            for r in donnees_v2._toutes_pages("timeonice", cay, True):
                e = rt.setdefault(str(r.get("gameId")), {}).setdefault(str(r["playerId"]), [None, 0.0, 0.0])
                e[1] = round((r.get("ppTimeOnIce") or 0) / 60, 2)
                e[2] = round((r.get("shTimeOnIce") or 0) / 60, 2)
            _ecrire(d / "rt.json", rt)
            etat["rt"] = True
        if saison != str(saison_courante) and not etat.get("meteo"):
            _ecrire(d / "meteo.json", _meteo(saison, cal))
            etat["meteo"] = True
        _ecrire(d / "etat.json", etat)
        pretes.append(saison)
    return pretes


# =================================================================================
# 2. Features (toujours avec SEULEMENT les matchs d'avant)
# =================================================================================

FEATS_J = ["def", "dom", "pj", "pj_prev", "t5", "t10", "t20", "t_sa", "t_prev", "toi5", "toi10", "toi_sa", "toi_prev",
           "t60_10", "t60_sa", "t60_prev", "att10", "att60_10", "att60_sa", "pp10", "pp_sa", "b10", "b_sa", "b_prev",
           "a10", "a_sa", "a_prev", "sh_pct", "t_std10", "mee10", "pim10", "repos", "b2b", "km", "fuseau",
           "o_ta10", "o_ta_sa", "o_ba_sa", "o_sv_sa", "o_mee10", "o_pim_sa", "o_repos", "o_b2b",
           "e_tf10", "e_bf_sa", "vs_t", "vs_n", "rink", "meteo", "mois"]
FEATS_M = ["h_bf10", "h_ba10", "h_tf10", "h_ta10", "h_bf_sa", "h_ba_sa", "h_sv_sa", "h_repos", "h_b2b", "h_km",
           "a_bf10", "a_ba10", "a_tf10", "a_ta10", "a_bf_sa", "a_ba_sa", "a_sv_sa", "a_repos", "a_b2b", "a_km",
           "h_mee10", "a_mee10", "h_pim_sa", "a_pim_sa", "rink", "meteo", "mois", "h_pj", "a_pj"]
NAN = float("nan")


def _moy(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else NAN


def _km(a, b):
    if a not in COORD or b not in COORD:
        return NAN
    (la1, lo1, _), (la2, lo2, _) = COORD[a], COORD[b]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    return 6371 * math.acos(min(1, max(-1, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl))))


class Etat:
    """Tout ce qu'on sait AVANT un match, mis à jour match par match, dans l'ordre des dates."""

    def __init__(self):
        self.j = defaultdict(lambda: deque(maxlen=20))     # pid -> derniers matchs (saison)
        self.j_sa = defaultdict(lambda: [0] * 9)            # pid -> totaux saison : pj toi tirs b a att pp mee pim
        self.j_prev = {}                                    # pid -> totaux saison passée
        self.e = defaultdict(lambda: deque(maxlen=10))      # équipe -> derniers matchs : bf ba tf ta mee pim
        self.e_sa = defaultdict(lambda: [0] * 7)            # équipe -> pj bf ba tf ta mee pim
        self.e_der = {}                                     # équipe -> (date, ville) du dernier match
        self.vs = defaultdict(lambda: [0, 0])               # (pid, adv) -> tirs, matchs (toutes saisons)
        self.rink = defaultdict(lambda: [0, 0, 0, 0])       # équipe -> tirs à domicile, matchs dom, tirs à l'ét., matchs ét.
        self.gard = defaultdict(lambda: [0, 0])             # gardien -> tirs contre, buts contre (saison + passée)
        self.saison = None

    def nouvelle_saison(self, saison):
        self.j_prev = {p: list(v) for p, v in self.j_sa.items()}
        self.j.clear(); self.j_sa.clear(); self.e.clear(); self.e_sa.clear(); self.rink.clear()
        self.saison = saison

    def _repos(self, t, d, ville):
        der = self.e_der.get(t)
        if not der:
            return NAN, 0, NAN, NAN
        jours = (date.fromisoformat(d) - date.fromisoformat(der[0])).days
        fz = abs(COORD.get(ville, (0, 0, 0))[2] - COORD.get(der[1], (0, 0, 0))[2]) if ville in COORD and der[1] in COORD else NAN
        return jours, 1 if jours == 1 else 0, _km(der[1], ville), fz

    def feats_joueur(self, pid, t, opp, dom, defense, d, meteo, gardien_adv=None):
        h = list(self.j[pid]); sa = self.j_sa[pid]; pv = self.j_prev.get(pid)
        der = lambda n, i: [x[i] for x in h[-n:]]
        def taux(n, i):
            xs = h[-n:]
            return sum(x[i] for x in xs) / len(xs) if xs else NAN
        def par60(n, i):
            xs = [x for x in h[-n:] if x[i] is not None]
            tt = sum(x[1] for x in xs)
            return sum(x[i] for x in xs) / tt * 60 if tt else NAN
        pj = sa[0]
        ville = t if dom else opp
        repos, b2b, km, fz = self._repos(t, d, ville)
        o = list(self.e[opp]); osa = self.e_sa[opp]
        ors, ob2b, _, _ = self._repos(opp, d, ville)
        e = list(self.e[t]); esa = self.e_sa[t]
        vs = self.vs[(pid, opp)]
        rk = self.rink[t if dom else opp]
        tirs10 = der(10, 2)
        return [defense, 1 if dom else 0, pj, pv[0] if pv else 0,
                taux(5, 2), taux(10, 2), taux(20, 2), sa[2] / pj if pj else NAN, pv[2] / pv[0] if pv and pv[0] else NAN,
                taux(5, 1), taux(10, 1), sa[1] / pj if pj else NAN, pv[1] / pv[0] if pv and pv[0] else NAN,
                par60(10, 2), sa[2] / sa[1] * 60 if sa[1] else NAN, pv[2] / pv[1] * 60 if pv and pv[1] else NAN,
                _moy(der(10, 5)), par60(10, 5), sa[5] / sa[1] * 60 if sa[1] and sa[5] else NAN,
                _moy(der(10, 6)), sa[6] / pj if pj else NAN,
                taux(10, 3), sa[3] / pj if pj else NAN, pv[3] / pv[0] if pv and pv[0] else NAN,
                taux(10, 4), sa[4] / pj if pj else NAN, pv[4] / pv[0] if pv and pv[0] else NAN,
                (sa[3] + (pv[3] if pv else 0) + 8) / (sa[2] + (pv[2] if pv else 0) + 80),
                (sum((x - sum(tirs10) / len(tirs10)) ** 2 for x in tirs10) / len(tirs10)) ** 0.5 if len(tirs10) >= 3 else NAN,
                _moy(der(10, 7)), _moy(der(10, 8)), repos, b2b, km, fz,
                _moy([x[3] for x in o]), osa[4] / osa[0] if osa[0] else NAN, osa[2] / osa[0] if osa[0] else NAN,
                1 - osa[2] / osa[4] if osa[4] else NAN, _moy([x[4] for x in o]), osa[6] / osa[0] if osa[0] else NAN,
                ors, ob2b,
                _moy([x[2] for x in e]), esa[1] / esa[0] if esa[0] else NAN,
                vs[0] / vs[1] if vs[1] else NAN, vs[1],
                (rk[0] / rk[1]) / (rk[2] / rk[3]) if rk[1] and rk[3] and rk[2] else NAN,
                meteo, int(d[5:7])]

    def feats_match(self, h, a, d, meteo):
        def eq(t, dom):
            e = list(self.e[t]); s = self.e_sa[t]
            repos, b2b, km, _ = self._repos(t, d, h)
            return [_moy([x[0] for x in e]), _moy([x[1] for x in e]), _moy([x[2] for x in e]), _moy([x[3] for x in e]),
                    s[1] / s[0] if s[0] else NAN, s[2] / s[0] if s[0] else NAN, 1 - s[2] / s[4] if s[4] else NAN,
                    repos, b2b, km]
        rk = self.rink[h]
        sh, sa = self.e_sa[h], self.e_sa[a]
        return (eq(h, True) + eq(a, False) +
                [_moy([x[4] for x in self.e[h]]), _moy([x[4] for x in self.e[a]]),
                 sh[6] / sh[0] if sh[0] else NAN, sa[6] / sa[0] if sa[0] else NAN,
                 (rk[0] / rk[1]) / (rk[2] / rk[3]) if rk[1] and rk[3] and rk[2] else NAN,
                 meteo, int(d[5:7]), sh[0], sa[0]])

    def ajouter(self, d, h, a, hs, as_, box, rt, gardiens):
        """Met l'état à jour avec un match FINI."""
        tot = {h: [0, 0, 0, 0], a: [0, 0, 0, 0]}   # tirs, buts, mises en échec, pim
        for pid, t, df, toi, tirs, b, pa, mee, pim, *_r in box:
            if t not in tot:
                continue
            r = rt.get(str(pid)) or [None, None, None]
            opp = a if t == h else h
            self.j[pid].append((d, toi, tirs, b, pa, r[0], r[1], mee, pim))
            s = self.j_sa[pid]
            for i, v in enumerate((1, toi, tirs, b, pa, r[0] or 0, r[1] or 0, mee or 0, pim or 0)):
                s[i] += v
            self.vs[(pid, opp)][0] += tirs; self.vs[(pid, opp)][1] += 1
            tot[t][0] += tirs; tot[t][1] += b; tot[t][2] += mee or 0; tot[t][3] += pim or 0
        if hs is None:
            hs, as_ = tot[h][1], tot[a][1]
        for t, o, gf, ga in ((h, a, hs, as_), (a, h, as_, hs)):
            self.e[t].append((gf, ga, tot[t][0], tot[o][0], tot[t][2], tot[t][3]))
            s = self.e_sa[t]
            for i, v in enumerate((1, gf, ga, tot[t][0], tot[o][0], tot[t][2], tot[t][3])):
                s[i] += v
            self.e_der[t] = (d, h)
            gd = gardiens.get(o)
            if gd:   # gardien de l'adversaire : tirs et buts qu'il a accordés (approx : toute l'équipe)
                self.gard[gd][0] += tot[t][0]; self.gard[gd][1] += gf
        rk = self.rink[h]
        rk[0] += tot[h][0] + tot[a][0]; rk[1] += 1
        rka = self.rink[a]
        rka[2] += tot[h][0] + tot[a][0]; rka[3] += 1


def construire(saisons_pretes, saison_courante, jour, stats_jeux_courant):
    """Toutes les lignes (joueur-match et match) des saisons prêtes, dans l'ordre des dates.
    Retourne (lignes_joueurs, lignes_matchs, état final pour prédire ce soir, météo de la saison courante)."""
    ordre = sorted(saisons_pretes)          # 20232024 < 20242025 < ... < saison courante
    LJ, LM = [], []
    E = Etat()
    for saison in ordre:
        d = DIR_HIST / saison
        cal = _lire(d / "calendrier.json", {}) if saison != str(saison_courante) else None
        box = _lire(d / "box.json", {})
        rt = _lire(d / "rt.json", {}) if saison != str(saison_courante) else {g: v.get("j", {}) for g, v in stats_jeux_courant.items()}
        met = _lire(d / "meteo.json", {})
        if cal is None:
            cal = _lire(d / "calendrier_courant.json", {})
        E.nouvelle_saison(saison)
        for gid, (dt, h, a, hs, as_) in sorted(cal.items(), key=lambda kv: kv[1][0]):
            if dt >= jour or gid not in box:
                continue
            b = box[gid]
            meteo = (met.get(h) or {}).get(dt, NAN)
            meteo = NAN if meteo is None else meteo
            gard = b.get("g", {})
            # Lignes AVANT de mettre l'état à jour (sinon le match se verrait lui-même)
            fm = E.feats_match(h, a, dt, meteo)
            if hs is not None:
                LM.append((dt, saison, gid, fm, hs, as_))
            for pid, t, df, toi, tirs, bu, pa, *_r in b["j"]:
                if t not in (h, a) or toi <= 0:
                    continue
                opp = a if t == h else h
                f = E.feats_joueur(pid, t, opp, t == h, df, dt, meteo, gard.get(opp))
                LJ.append((dt, saison, gid, pid, f, tirs, bu, pa))
            E.ajouter(dt, h, a, hs, as_, b["j"], rt.get(gid, {}), gard)
    return LJ, LM, E


# =================================================================================
# 3. Entraîner, valider honnêtement, prédire
# =================================================================================

def _cible(cat, tirs, b, a):
    stat, seuil = CATS_J[cat]
    v = {"tirs": tirs, "buts": b, "passes": a, "pts": b + a}[stat]
    return 1 if v >= seuil else 0


def _modele(Clf, n):
    # Prudent : arbres peu profonds, beaucoup d'exemples par feuille, arrêt hâtif.
    return Clf(learning_rate=0.05, max_iter=400, max_leaf_nodes=24, min_samples_leaf=max(40, n // 2000),
               l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)


class _Ajuste:
    """Un modèle + les colonnes vides à l'entraînement (ex. la météo pas encore récoltée).
    scikit-learn plante sur une colonne 100 % vide : on la met à 0, et pareil à la prédiction."""

    def __init__(self, modele, X, y):
        import numpy as np
        self.vides = np.all(np.isnan(X), axis=0)
        Xc = X.copy(); Xc[:, self.vides] = 0
        self.m = modele.fit(Xc, y)

    def _prep(self, X):
        Xc = X.copy(); Xc[:, self.vides] = 0
        return Xc

    def predict_proba(self, X):
        return self.m.predict_proba(self._prep(X))

    def predict(self, X):
        return self.m.predict(self._prep(X))


def _brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys) if ys else None


def _semaine(d):
    x = date.fromisoformat(d)
    return (x - timedelta(days=x.weekday())).isoformat()


def entrainer(LJ, LM, saison_courante, historique, principales, juger=True, retro_max_semaines=2):
    """Entraîne les modèles de production (toutes les données), écrit les prédictions honnêtes
    des soirs passés (semaine par semaine) et, si `juger`, mesure chaque catégorie contre la
    recette en service sur les 30 % de soirs les plus récents, jamais vus."""
    Clf, Reg = _sklearn()
    import numpy as np
    sc = str(saison_courante)
    out = {"rapports": {}, "retro": {}, "retro_m": {}, "n_lignes": len(LJ), "n_matchs": len(LM), "modeles": None}
    if len(LJ) < MIN_LIGNES:
        out["msg"] = f"pas encore assez de lignes pour apprendre ({len(LJ)})"
        return out
    X = np.array([r[4] for r in LJ], dtype=float)
    dates = [r[0] for r in LJ]
    ys = {cat: np.array([_cible(cat, r[5], r[6], r[7]) for r in LJ]) for cat in CATS_J}
    cnt = {"tirs": np.array([r[5] for r in LJ]), "buts": np.array([r[6] for r in LJ]), "passes": np.array([r[7] for r in LJ])}
    XM = np.array([r[3] for r in LM], dtype=float) if LM else None
    dM = [r[0] for r in LM]
    yM = {"vic": np.array([1 if r[4] > r[5] else 0 for r in LM]),
          "o55": np.array([1 if r[4] + r[5] > 5.5 else 0 for r in LM]),
          "o65": np.array([1 if r[4] + r[5] > 6.5 else 0 for r in LM])} if LM else {}

    # --- Production : tout ce qui précède ce soir ---
    mods = {"clf": {}, "reg": {}, "m": {}}
    for stat in ("tirs", "buts", "passes"):
        mods["reg"][stat] = _Ajuste(Reg(loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=24,
                                        min_samples_leaf=60, early_stopping=True, random_state=0), X, cnt[stat])
    for cat in CATS_J:
        mods["clf"][cat] = _Ajuste(_modele(Clf, len(X)), X, ys[cat])
    if XM is not None and len(XM) >= MIN_MATCHS:
        for k in yM:
            mods["m"][k] = _Ajuste(_modele(Clf, len(XM)), XM, yM[k])
    out["modeles"] = mods

    # --- Walk-forward : semaine par semaine, modèle entraîné seulement sur l'avant ---
    jours_sc = sorted({r[0] for r in LJ if r[1] == sc})
    deja = {h.get("ml_semaine") for h in historique if h.get("ml_semaine")}
    semaines = sorted({_semaine(d) for d in jours_sc})
    for w in [w for w in semaines if w not in deja][:retro_max_semaines]:
        fin = (date.fromisoformat(w) + timedelta(days=7)).isoformat()
        tr = np.array([d < w for d in dates]); va = np.where(np.array([w <= d < fin for d in dates]))[0]
        preds_w, preds_m = {}, {}
        if tr.sum() >= MIN_LIGNES and len(va):
            for cat in CATS_J:
                ps = _Ajuste(_modele(Clf, int(tr.sum())), X[tr], ys[cat][tr]).predict_proba(X[va])[:, 1]
                for i, ri in enumerate(va):
                    preds_w.setdefault((LJ[ri][0], LJ[ri][3]), {})[f"{cat}_ml"] = {MARCHE_DE[cat]: {"p": round(float(ps[i]), 4)}}
            if XM is not None:
                trm = np.array([d < w for d in dM]); vam = np.where(np.array([w <= d < fin for d in dM]))[0]
                if trm.sum() >= MIN_MATCHS and len(vam):
                    pm = {k: _Ajuste(_modele(Clf, int(trm.sum())), XM[trm], yM[k][trm]).predict_proba(XM[vam])[:, 1] for k in yM}
                    for i, ri in enumerate(vam):
                        preds_m[(LM[ri][0], LM[ri][2])] = {
                            "vic_ml": {"p_home": round(float(pm["vic"][i]), 4)},
                            "tot_ml": {"p_over": {"5.5": round(float(pm["o55"][i]), 4), "6.5": round(float(pm["o65"][i]), 4)}}}
        out["retro"][w] = preds_w
        out["retro_m"][w] = preds_m

    # --- Juge honnête (hebdo) ---
    if juger and len(jours_sc) >= 6:
        coupe = jours_sc[int(len(jours_sc) * 0.7)]
        tr = np.array([d < coupe for d in dates])
        va_i = [i for i, d in enumerate(dates) if d >= coupe and LJ[i][1] == sc]
        stock, stock_m = {}, {}
        for h in historique:
            if h["date"] >= coupe:
                for e in h["entries"]:
                    if e.get("played"):
                        stock[(h["date"], e["pid"])] = e.get("preds", {})
                for g in h.get("games", []):
                    stock_m[(h["date"], str(g["gameId"]))] = g.get("preds", {})
        for cat in CATS_J:
            champ = principales.get(cat, "v1")
            ps = _Ajuste(_modele(Clf, int(tr.sum())), X[tr], ys[cat][tr]).predict_proba(X[va_i])[:, 1]
            a, b, y = [], [], []
            for i, ri in enumerate(va_i):
                pv = (stock.get((LJ[ri][0], LJ[ri][3])) or {}).get(champ, {}).get(MARCHE_DE[cat])
                if pv:
                    a.append(float(ps[i])); b.append(pv["p"]); y.append(int(ys[cat][ri]))
            b_ml, b_ch = _brier(a, y), _brier(b, y)
            out["rapports"][cat] = {"champion": champ, "n": len(y), "pos": int(sum(y)), "coupe": coupe,
                                    "gain": (b_ch - b_ml) / b_ch if b_ch and b_ml is not None else None}
        if XM is not None:
            trm = np.array([d < coupe for d in dM]); vam = [i for i, d in enumerate(dM) if d >= coupe and LM[i][1] == sc]
            if trm.sum() >= MIN_MATCHS and vam:
                pm = {k: _Ajuste(_modele(Clf, int(trm.sum())), XM[trm], yM[k][trm]).predict_proba(XM[vam])[:, 1] for k in yM}
                for cat, cles in (("vic", ("vic",)), ("tot", ("o55", "o65"))):
                    champ = principales.get(cat, "v1")
                    a, b, y = [], [], []
                    for i, ri in enumerate(vam):
                        pv = (stock_m.get((LM[ri][0], LM[ri][2])) or {}).get(champ)
                        if not pv:
                            continue
                        for k in cles:
                            pc = pv.get("p_home") if k == "vic" else (pv.get("p_over") or {}).get("5.5" if k == "o55" else "6.5")
                            if pc is not None:
                                a.append(float(pm[k][i])); b.append(pc); y.append(int(yM[k][ri]))
                    b_ml, b_ch = _brier(a, y), _brier(b, y)
                    out["rapports"][cat] = {"champion": champ, "n": len(y), "pos": int(sum(y)), "coupe": coupe,
                                            "gain": (b_ch - b_ml) / b_ch if b_ch and b_ml is not None else None}
    return out


def predire_soir(res, E, jour, entrees, games, meteo_soir):
    """Ajoute les prédictions ML de ce soir aux joueurs et aux matchs (en place)."""
    mods = res.get("modeles")
    if not mods:
        return 0
    import numpy as np
    if entrees:
        Xs = np.array([E.feats_joueur(e["pid"], e["team"], e["opp"], e["home"], 1 if e["pos"] == "D" else 0, jour,
                                      _meteo_de(meteo_soir, e["team"] if e["home"] else e["opp"], jour)) for e in entrees], dtype=float)
        lam = {s: mods["reg"][s].predict(Xs) for s in mods["reg"]}
        for cat, m in mods["clf"].items():
            ps = m.predict_proba(Xs)[:, 1]
            stat = CATS_J[cat][0]
            for i, e in enumerate(entrees):
                l = float(lam["buts"][i] + lam["passes"][i]) if stat == "pts" else float(lam[stat][i])
                e["preds"][f"{cat}_ml"] = {MARCHE_DE[cat]: {"p": round(float(ps[i]), 4), "lam": round(l, 3)}}
    if games and mods["m"]:
        XM = np.array([E.feats_match(g["home"], g["away"], jour, _meteo_de(meteo_soir, g["home"], jour)) for g in games], dtype=float)
        pm = {k: m.predict_proba(XM)[:, 1] for k, m in mods["m"].items()}
        for i, g in enumerate(games):
            g["preds"]["vic_ml"] = {"p_home": round(float(pm["vic"][i]), 4)}
            g["preds"]["tot_ml"] = {"p_over": {"5.5": round(float(pm["o55"][i]), 4), "6.5": round(float(pm["o65"][i]), 4)}}
    return len(entrees)


def _meteo_de(meteo, t, jour):
    v = (meteo.get(t) or {}).get(jour) if meteo else None
    return NAN if v is None else v


def ecrire_retro(res, fichiers_pred, lire, ecrire):
    """Colle les prédictions walk-forward dans les fichiers de prédictions des soirs passés."""
    n = 0
    for w in set(res.get("retro", {})) | set(res.get("retro_m", {})):
        pj, pm = res["retro"].get(w, {}), res.get("retro_m", {}).get(w, {})
        fin = (date.fromisoformat(w) + timedelta(days=7)).isoformat()
        for f in fichiers_pred:
            if not (w <= f.stem < fin):
                continue
            data = lire(f, None)
            if not data:
                continue
            for e in data["entries"]:
                p = pj.get((data["date"], e["pid"]))
                if p:
                    e.setdefault("preds", {}).update(p); n += 1
            for g in data.get("games", []):
                p = pm.get((data["date"], str(g["gameId"])))
                if p:
                    g.setdefault("preds", {}).update(p)
            data["ml_semaine"] = w
            ecrire(f, data)
    return n
