"""
Le labo : l'algo qui s'améliore tout seul, AVEC des garde-fous.

1. DÉTECTEUR DE PATTERNS (à chaque calcul) : segments où la version principale se trompe
   systématiquement (défenseurs, back-to-back...), quand l'écart est trop gros pour être
   de la chance (z >= 2.5). C'est le « reverse engineering » : on voit OÙ l'algo se trompe.

2. RECALCUL HEBDO PAR CATÉGORIE (voir section 3) : une recette par catégorie (tirs 1+ à
   4+, but, passe, victoire, totaux), rescannée chaque semaine avec toute la saison, gardée
   seulement si elle fait mieux sur des soirs qu'elle n'a jamais vus pendant le scan.

Limite assumée : le scan ajuste des poids, il n'invente pas de nouvelles infos. Les gros
gains viennent de nouvelles idées (alignements, gardien partant...), ajoutées à la main.
"""

import json
import math
import random
from datetime import date, datetime, timedelta, timezone

from modeles import FAMILLES, V1, VERSIONS

# Famille de modèle en cours de tuning (v1 ou v2) : fixée par cycle() selon la principale
_FAM = FAMILLES["v1"]
# Premier soir de VALIDATION du dernier tuner/calibreur : les soirs d'avant ont servi à
# l'entraîner, donc ils ne comptent pas pour le juger.
DERNIERE_COUPE = None

# --- Seuils (les tests les baissent pour rouler sur une petite fausse ligue) ------
MIN_JOUEURS = 2000          # prédictions de joueurs gradées avec features avant d'oser tuner
MIN_MATCHS = 120            # idem pour les matchs
MIN_VAL_JOUEURS = 600       # taille minimale du jeu de validation (par marché joueur)
MIN_VAL_MATCHS = 40
GAIN_MIN = 0.005            # amélioration relative du Brier exigée en validation (0.5 %)
PIRE_MAX = 0.003            # dégradation tolérée sur un autre marché (0.3 %)
PROMO_MIN_TIRS = 1500       # prédictions communes (même joueur, même soir) avant de promouvoir
PROMO_GAIN = 0.005
Z_PATTERN = 2.5

# Espace de recherche : (paramètre, min, max, type de pas)
# "mult" = essais à ×0.6 … ×1.6 ; "add" = petits pas additifs (facteurs proches de 1)
ESPACE = {
    "tirs": [("w_l10", 0.5, 4, "mult"), ("w_prev", 0.0, 1.5, "mult"), ("pseudo_min", 20, 600, "mult"),
             ("toi_recent", 0.0, 1.0, "add"), ("opp_reg_gp", 3, 60, "mult"), ("home", 0.95, 1.10, "add"),
             ("away", 0.90, 1.05, "add"), ("b2b", 0.85, 1.05, "add"), ("vs_reg_gp", 3, 60, "mult"),
             ("nb_k", 3, 200, "mult"), ("scale_tirs", 0.85, 1.15, "add"), ("scale_tirs_D", 0.80, 1.20, "add")],
    "buts": [("sh_pseudo_tirs", 10, 400, "mult"), ("sv_reg_gp", 3, 60, "mult"), ("scale_buts", 0.80, 1.20, "add")],
    "passes": [("pa_pseudo_min", 50, 1500, "mult"), ("scale_passes", 0.80, 1.20, "add")],
    "match": [("m_w_l10", 0.5, 4, "mult"), ("m_w_prev", 0.0, 1.5, "mult"), ("m_pseudo_gp", 2, 40, "mult"),
              ("m_home", 1.0, 1.08, "add"), ("m_away", 0.92, 1.0, "add"), ("m_h2h_reg", 2, 40, "mult"),
              ("scale_match", 0.85, 1.15, "add"), ("m_p_ot_home", 0.45, 0.60, "add")],
}
# Paramètres en plus, propres à la v2
ESPACE_V2 = {
    "tirs": [("k_pp", 0.0, 4.0, "mult"), ("w_att", 0.0, 1.0, "mult"), ("pen_exp", 0.0, 2.0, "mult"),
             ("score_s", 0.0, 0.4, "mult"), ("w_mod_joueur", 0.1, 1.0, "mult"), ("vig_props", 1.0, 1.15, "add"),
             ("w_rink", 0.0, 1.5, "mult"), ("prior_toi_exp", 0.0, 1.5, "mult"),
             ("cal_t_a", 0.7, 1.3, "add"), ("cal_t_b", -0.3, 0.3, "add")],
    "buts": [("cal_bu_a", 0.7, 1.3, "add"), ("cal_bu_b", -0.3, 0.3, "add")],
    "passes": [("k_pp_passes", 0.0, 4.0, "mult"), ("cal_pa_a", 0.7, 1.3, "add"), ("cal_pa_b", -0.3, 0.3, "add")],
    "match": [("cal_a", 0.6, 1.6, "mult"), ("cal_b", -0.3, 0.3, "add"), ("mt_w_buts", 0.0, 1.0, "mult"),
              ("mt_pseudo_gp", 2.0, 30.0, "mult"), ("mt_sh_pseudo", 100.0, 2000.0, "mult"), ("w_mod_match", 0.1, 1.0, "mult")],
}
SEUILS = {"tirs": 2, "buts": 1, "passes": 1}
NOMS = {"tirs": "Tirs (1+ à 4+)", "buts": "But", "passes": "Passe", "match": "Matchs"}


# =================================================================================
# Données
# =================================================================================

def _tuple_caps(P):
    P = dict(P)
    P["vs_cap"], P["m_h2h_cap"] = tuple(P["vs_cap"]), tuple(P["m_h2h_cap"])
    return P


def donnees(historique):
    """Lignes gradées qui ont des features. Joueurs : (date, f, résultats, preds).
    Matchs : (date, f, (buts dom., buts vis.), preds)."""
    J, M = [], []
    for fic in historique:
        d = fic["date"]
        for e in fic["entries"]:
            if e.get("played") and isinstance(e.get("result"), dict) and e.get("f"):
                f = e["f"] if "s10" in e["f"] else dict(e["f"], s10=e.get("l10_shots"))
                J.append((d, f, e["result"], e.get("preds", {})))
        for g in fic.get("games", []):
            if g.get("result") and g.get("f"):
                M.append((d, g["f"], (g["result"]["home"], g["result"]["away"]), g.get("preds", {})))
    return J, M


# Pour les tirs, le tuner optimise la moyenne des 4 seuils : c'est la seule façon
# d'ajuster correctement la dispersion (nb_k). Avec juste le 2+, une courbe trop
# étroite ou trop large peut donner le même 2+ et se tromper sur 1+ et 4+.
SEUILS_TUNER = {"tirs": {"tirs1": 1, "tirs": 2, "tirs3": 3, "tirs4": 4}, "buts": {"buts": 1}, "passes": {"passes": 1}}


def _brier_joueurs(rows, P, marche):
    stat = marche
    s = n = 0
    for _, f, r, _ in rows:
        if stat not in r:
            continue
        sortie = _FAM["joueur"](f, P)
        for m, seuil in SEUILS_TUNER[marche].items():
            y = 1 if r[stat] >= seuil else 0
            s += (sortie[m]["p"] - y) ** 2; n += 1
    return s / n if n else None


def _brier_matchs(rows, P):
    """Somme des Brier victoire + total 5.5 (les deux décisions qu'on regarde)."""
    s = n = 0
    for _, f, (h, a), _ in rows:
        r = _FAM["match"](f, P)
        s += (r["p_home"] - (1 if h > a else 0)) ** 2 + (r["p_over"]["5.5"] - (1 if h + a > 5.5 else 0)) ** 2
        n += 1
    return s / n if n else None


def _perte(groupe, rows, P):
    return _brier_matchs(rows, P) if groupe == "match" else _brier_joueurs(rows, P, groupe)


# =================================================================================
# 1. Détecteur de patterns
# =================================================================================

def _segments_joueur(f):
    """Étiquettes de segments pour un joueur (une ligne peut être dans plusieurs)."""
    o, lg = f["op"], f["lg"]
    f_opp = (o[0] * o[1] + 15 * o[2]) / (o[0] + 15) / lg[0]
    toi = f["tsa"] if f["tsa"] is not None else (f["pv"][3] / f["pv"][4] if f["pv"][4] else lg[5])
    seg = ["Défenseurs" if f["pos"] == "D" else "Avants",
           "À domicile" if f["home"] else "Sur la route",
           "Temps de glace 20 min et +" if toi >= 20 else "Temps de glace 16-20 min" if toi >= 16
           else "Temps de glace 12-16 min" if toi >= 12 else "Temps de glace moins de 12 min",
           "Contre une équipe qui accorde beaucoup de tirs" if f_opp >= 1.06
           else "Contre une équipe qui accorde peu de tirs" if f_opp <= 0.94 else None,
           "Début de saison (moins de 5 matchs)" if f["n"] < 5 else None]
    if f["b2b"]:
        seg.append("2e match en 2 soirs")
    return [s for s in seg if s]


def patterns(J, M, version):
    """Segments où `version` se trompe systématiquement. Retourne une liste triée."""
    out = []
    for marche in ("tirs", "buts", "passes"):
        agg = {}
        for _, f, r, preds in J:
            pv = preds.get(version, {}).get(marche)
            if not pv or marche not in r:
                continue
            y = 1 if r[marche] >= SEUILS[marche] else 0
            for s in _segments_joueur(f):
                a = agg.setdefault(s, [0, 0.0, 0.0, 0.0])
                a[0] += 1; a[1] += pv["p"]; a[2] += y; a[3] += pv["p"] * (1 - pv["p"])
        out += _signaler(agg, marche, 150)
    agg = {}
    for _, f, (h, a), preds in M:
        pv = preds.get(version)
        if not pv:
            continue
        segs = ["Favori à domicile" if pv["p_home"] >= 0.5 else "Favori sur la route"]
        if f["h"]["b2b"] or f["a"]["b2b"]:
            segs.append("Une équipe en 2e match en 2 soirs")
        for s in segs:
            x = agg.setdefault(s, [0, 0.0, 0.0, 0.0])
            x[0] += 1; x[1] += pv["p_home"]; x[2] += 1 if h > a else 0; x[3] += pv["p_home"] * (1 - pv["p_home"])
    out += _signaler(agg, "victoire (domicile)", 40)
    return sorted(out, key=lambda x: -abs(x["z"]))[:10]


def _signaler(agg, marche, n_min):
    res = []
    for seg, (n, sp, sy, var) in agg.items():
        if n < n_min or var <= 0:
            continue
        predit, reel = sp / n, sy / n
        # z = écart / écart-type attendu par pure chance. |z| >= 2.5 : ~1 % de chance que ce soit du hasard.
        z = (sy - sp) / math.sqrt(var)
        if abs(z) >= Z_PATTERN:
            res.append({"marche": marche, "segment": seg, "n": n, "predit": predit, "reel": reel, "z": z,
                        "sens": "sous-estimé" if reel > predit else "surestimé"})
    return res


# =================================================================================
# 2. Tuner
# =================================================================================

def _candidats(val, lo, hi, pas):
    if pas == "mult":
        vals = [val * k for k in (0.6, 0.8, 0.9, 1.1, 1.25, 1.6)] if val else [lo + (hi - lo) * k for k in (0.1, 0.25, 0.5)]
    else:
        vals = [val + d for d in (-0.04, -0.02, -0.01, 0.01, 0.02, 0.04)]
    return sorted({min(max(v, lo), hi) for v in vals} - {val})


def _descente(groupe, rows, P, passes=2, espace=None):
    """Descente par coordonnées : un paramètre à la fois, on garde ce qui baisse la perte."""
    P = dict(P)
    meilleur = _perte(groupe, rows, P)
    if espace is None:
        espace = ESPACE[groupe] + (ESPACE_V2[groupe] if _FAM is FAMILLES["v2"] else [])
    for _ in range(passes):
        ameliore = False
        for nom, lo, hi, pas in espace:
            base = P.get(nom, 1.0)
            for v in _candidats(base, lo, hi, pas):
                essai = dict(P, **{nom: v})
                perte = _perte(groupe, rows, essai)
                if perte is not None and perte < meilleur - 1e-7:
                    meilleur, P, ameliore = perte, essai, True
        if not ameliore:
            break
    return P


def tuner(J, M, P_ref, aujourdhui):
    """Cherche une meilleure variante de P_ref. Retourne (nouveaux params ou None, rapport)."""
    # Plafond : les 40 000 prédictions de joueurs les plus récentes. Garde le tuning
    # sous ~3 min en fin de saison, et le hockey de février ressemble plus à mars qu'à octobre.
    J = sorted(J, key=lambda r: r[0])[-40000:]
    dates = sorted({d for d, *_ in J} | {d for d, *_ in M})
    if len(dates) < 4:
        return None, "Pas assez de soirs pour séparer entraînement et validation."
    coupe = dates[int(len(dates) * 0.7)]
    global DERNIERE_COUPE
    DERNIERE_COUPE = coupe
    Jt, Jv = [r for r in J if r[0] < coupe], [r for r in J if r[0] >= coupe]
    Mt, Mv = [r for r in M if r[0] < coupe], [r for r in M if r[0] >= coupe]

    P = _tuple_caps(P_ref)
    gains, details = {}, []
    # Ordre : tirs d'abord (les buts en dépendent), puis buts, passes, matchs.
    for groupe in ("tirs", "buts", "passes", "match"):
        t, v = (Mt, Mv) if groupe == "match" else (Jt, Jv)
        n_val = len(v)
        assez = n_val >= (MIN_VAL_MATCHS if groupe == "match" else MIN_VAL_JOUEURS) and len(t) >= n_val
        if not assez:
            details.append(f"{NOMS[groupe]} : pas assez de données ({len(t)} entr. / {n_val} valid.)")
            continue
        cand = _descente(groupe, t, P)
        avant, apres = _perte(groupe, v, P), _perte(groupe, v, cand)
        gain = (avant - apres) / avant if avant else 0
        if gain >= GAIN_MIN:
            P = cand
            gains[groupe] = gain
            details.append(f"{NOMS[groupe]} : {gain * 100:.1f} % mieux en validation")
        else:
            details.append(f"{NOMS[groupe]} : rien de mieux qui tient en validation ({gain * 100:+.1f} %)")

    if not gains:
        return None, " · ".join(details)

    # Garde-fou final : la variante complète ne doit empirer aucun marché en validation.
    ref = _tuple_caps(P_ref)
    for groupe in ("tirs", "buts", "passes", "match"):
        v = Mv if groupe == "match" else Jv
        if not v:
            continue
        a, b = _perte(groupe, v, ref), _perte(groupe, v, P)
        if a and (b - a) / a > PIRE_MAX:
            return None, " · ".join(details) + f" · REJETÉ : {NOMS[groupe]} empire de {(b - a) / a * 100:.1f} %"
    return P, " · ".join(details)


# =================================================================================
# 3. Recalcul hebdomadaire PAR CATÉGORIE (24 octobre)
# =================================================================================
#
# Pas un concours de versions : on cherche LA meilleure recette pour chaque catégorie,
# avec toutes les données de la saison. Une fois par semaine :
#   1. Pour chacune des 8 catégories, scan de tous les paramètres (descente par
#      coordonnées) qui minimise l'erreur (Brier) de CETTE catégorie seulement.
#   2. Test honnête : on refait le même scan sur les 70 % plus vieux soirs, et on mesure
#      sur les 30 % plus récents, jamais vus. La nouvelle recette remplace la version en
#      service seulement si elle fait mieux sur ces soirs-là. Sinon on se mentirait : un
#      scan « apprend » aussi le bruit des soirs qu'il voit (le 23 oct. : +0.37 % sur ce
#      qu'il avait vu, +0.10 % pour de vrai).
#   3. Une version gagnante s'appelle tirs2_v5, but_v6, etc. Elle est notée au Suivi
#      seulement à partir de sa création (elle a vu tout ce qui vient avant).
# Les nouvelles IDÉES (nouvel ingrédient, autre façon de voir) s'ajoutent à la main, quand
# on en trouve une. Pas de versions pour le fun.

JOURS_RECALCUL = 7
REFERENCE = "v1"
GAIN_HEBDO = 0.001          # la nouvelle recette doit faire au moins 0.1 % mieux sur les soirs jamais vus
MIN_LIGNES = {"joueur": 2000, "match": 100}
PREMIERE_VERSION = 5        # les versions par catégorie commencent à v5 (après les v4 de l'ancien moteur)

# Catégorie -> (type, marchés)
CATS = {"tirs1": ("joueur", ("tirs1",)), "tirs2": ("joueur", ("tirs",)), "tirs3": ("joueur", ("tirs3",)),
        "tirs4": ("joueur", ("tirs4",)), "but": ("joueur", ("buts",)), "passe": ("joueur", ("passes",)),
        "vic": ("match", ("victoire",)), "tot": ("match", ("5.5", "6.5"))}
NOMS_CAT = {"tirs1": "Tirs 1+", "tirs2": "Tirs 2+", "tirs3": "Tirs 3+", "tirs4": "Tirs 4+", "but": "But",
            "passe": "Passe", "vic": "Victoire", "tot": "Totaux 5.5 / 6.5"}
MARCHES_L = {"tirs1": ("tirs", 1), "tirs": ("tirs", 2), "tirs3": ("tirs", 3), "tirs4": ("tirs", 4),
             "buts": ("buts", 1), "passes": ("passes", 1)}

# Paramètres scannés pour chaque catégorie (ceux qui la touchent vraiment)
_DISP = [("w_disp", 0.0, 1.0, "mult")]
ESPACE_CAT = {
    "tirs": ESPACE["tirs"] + ESPACE_V2["tirs"] + _DISP,
    "but": ESPACE["tirs"] + ESPACE_V2["tirs"] + ESPACE["buts"] + ESPACE_V2["buts"],
    "passe": [e for e in ESPACE["tirs"] if e[0] in ("w_l10", "w_prev", "toi_recent", "pseudo_min")]
             + ESPACE["passes"] + ESPACE_V2["passes"] + [e for e in ESPACE_V2["tirs"] if e[0] in ("k_pp", "w_mod_joueur", "prior_toi_exp")],
    "match": ESPACE["match"] + ESPACE_V2["match"],
}


def _espace_cat(cat):
    if cat.startswith("tirs"):
        return ESPACE_CAT["tirs"]
    return ESPACE_CAT.get(cat) or ESPACE_CAT["match"]


def _pertes_match(pv, h, a):
    """Brier victoire + total 5.5 + total 6.5 d'une prédiction de match, par marché."""
    po = pv.get("p_over") or {}
    out = {}
    if "p_home" in pv:
        out["victoire"] = (pv["p_home"] - (1 if h > a else 0)) ** 2
    for ligne in ("5.5", "6.5"):
        if ligne in po:
            out[ligne] = (po[ligne] - (1 if h + a > float(ligne) else 0)) ** 2
    return out


def _perte_cat(cat, rows, P):
    """Brier moyen de la recette P sur UNE catégorie."""
    typ, marches = CATS[cat]
    fam = FAMILLES["v2"]
    s = n = 0
    if typ == "joueur":
        for _, f, r, _ in rows:
            o = fam["joueur"](f, P)
            for m in marches:
                stat, seuil = MARCHES_L[m]
                if stat in r and m in o:
                    s += (o[m]["p"] - (1 if r[stat] >= seuil else 0)) ** 2; n += 1
    else:
        for _, f, (h, a), _ in rows:
            q = _pertes_match(fam["match"](f, P), h, a)
            for m in marches:
                if m in q:
                    s += q[m]; n += 1
    return s / n if n else None


def _perte_stockee(cat, rows, vid):
    """Brier d'une version déjà en service, avec SES prédictions sauvées (mêmes lignes)."""
    typ, marches = CATS[cat]
    s = n = 0
    for row in rows:
        pv = row[3].get(vid)
        if not pv:
            continue
        if typ == "joueur":
            r = row[2]
            for m in marches:
                stat, seuil = MARCHES_L[m]
                if stat in r and m in pv:
                    s += (pv[m]["p"] - (1 if r[stat] >= seuil else 0)) ** 2; n += 1
        else:
            h, a = row[2]
            q = _pertes_match(pv, h, a)
            for m in marches:
                if m in q:
                    s += q[m]; n += 1
    return (s / n if n else None), n


def _scanner(cat, rows, P, passes=2):
    """Descente par coordonnées sur les paramètres de la catégorie."""
    P = _tuple_caps(P)
    meilleur = _perte_cat(cat, rows, P)
    for _ in range(passes):
        ameliore = False
        for nom, lo, hi, pas in _espace_cat(cat):
            base = P.get(nom, 1.0)
            for v in _candidats(base, lo, hi, pas):
                essai = dict(P, **{nom: v})
                perte = _perte_cat(cat, rows, essai)
                if perte is not None and perte < meilleur - 1e-7:
                    meilleur, P, ameliore = perte, essai, True
        if not ameliore:
            break
    return P


def _params_de(vid):
    """Paramètres complets d'une version (famille v2). v1 (code gelé) -> son équivalent
    paramétrique dans la famille v2 (ajouts v2 coupés)."""
    spec = VERSIONS.get(vid) or {}
    if "params" in spec and spec.get("famille") == "v2":
        return dict(spec["params"])
    base = dict(FAMILLES["v2"]["P"])
    base.update({k: 0.0 for k in ("w_att", "k_pp", "pen_exp", "score_s", "k_pp_passes")})
    base["w_mod_joueur"] = base["w_mod_match"] = 1.0     # comme v1 : pas de mélange avec le marché
    if "params" in spec:
        base.update({k: v for k, v in spec["params"].items() if k in base})
    return base


def recalculer(cat, J, M, champion, aujourdhui, numero):
    """Scan hebdo d'une catégorie. Retourne (nouvelle version ou None, rapport)."""
    typ, _ = CATS[cat]
    rows = sorted(J if typ == "joueur" else M, key=lambda r: r[0])
    if typ == "joueur":
        rows = rows[-40000:]   # plafond : garde le scan sous quelques minutes en fin de saison
    if len(rows) < MIN_LIGNES[typ]:
        return None, {"cat": cat, "msg": f"pas assez de données ({len(rows)} lignes)"}
    dates = sorted({r[0] for r in rows})
    coupe = dates[int(len(dates) * 0.7)]
    train, val = [r for r in rows if r[0] < coupe], [r for r in rows if r[0] >= coupe]
    P0 = _params_de(champion)
    # 1. Le test honnête : scan sur le vieux, mesure sur le récent jamais vu
    Pcv = _scanner(cat, train, P0)
    val_c = [r for r in val if r[3].get(champion)]
    b_champ, n_val = _perte_stockee(cat, val_c, champion)
    b_cv = _perte_cat(cat, val_c, Pcv)
    if not b_champ or b_cv is None:
        return None, {"cat": cat, "msg": "pas de soirs de validation comparables"}
    gain_cv = (b_champ - b_cv) / b_champ
    rapport = {"cat": cat, "champion": champion, "gain_honnete": gain_cv, "n_val": n_val, "coupe": coupe}
    if gain_cv < GAIN_HEBDO:
        rapport["msg"] = f"{champion} reste : la meilleure recette trouvée fait {gain_cv * 100:+.2f} % sur les soirs jamais vus"
        return None, rapport
    # 2. Gagnant : on refait le scan avec TOUTE la saison (départ du résultat du test)
    P = _scanner(cat, rows, Pcv)
    b_tout, _ = _perte_stockee(cat, [r for r in rows if r[3].get(champion)], champion)
    b_new = _perte_cat(cat, [r for r in rows if r[3].get(champion)], P)
    rapport["gain_saison"] = (b_tout - b_new) / b_tout if b_tout else None
    base = FAMILLES["v2"]["P"]
    norm = lambda v: list(v) if isinstance(v, tuple) else v
    diff = {k: norm(v) for k, v in P.items() if norm(base.get(k, 1.0 if k.startswith("scale") else None)) != norm(v)}
    vid = f"{cat}_v{numero}"
    rapport["msg"] = (f"{vid} remplace {champion} : {gain_cv * 100:+.2f} % sur les soirs jamais vus"
                      + (f", {rapport['gain_saison'] * 100:+.2f} % sur toute la saison (optimiste)" if rapport["gain_saison"] is not None else ""))
    version = {"id": vid, "cat": cat, "famille": "v2", "cree": aujourdhui, "juge_depuis": aujourdhui,
               "base": champion, "params": diff, "retiree": False, "rapport": rapport,
               "desc": f"{NOMS_CAT[cat]} : scan du {aujourdhui} sur toute la saison. "
                       f"{gain_cv * 100:+.2f} % vs {champion} sur des soirs jamais vus pendant le scan."}
    return version, rapport


def cycle(historique, chemin_etat, versions_actives, aujourdhui=None, force=False):
    """Patterns + recalcul hebdo par catégorie. Met à jour data/versions_auto.json.
    Retourne (état, True si une version a changé)."""
    aujourdhui = aujourdhui or date.today().isoformat()
    try:
        etat = json.loads(open(chemin_etat, encoding="utf-8").read())
    except (FileNotFoundError, ValueError):
        etat = {}
    etat.setdefault("versions", []); etat.setdefault("journal", [])
    nouvelle = False

    # --- Migration (moteur 5, 24 oct.) : fin du concours de générations. Toutes les
    # versions de l'ancien moteur arrêtent de prédire (leur historique reste) ; chaque
    # catégorie repart de v1 et le recalcul hebdo roule tout de suite. ---
    if etat.get("moteur", 0) < 5:
        for v in etat["versions"]:
            v["retiree"] = True
        etat["retirees"] = sorted(set(etat.get("retirees", [])) | {v for v, s in VERSIONS.items()
                                                                     if s.get("generation") in (2, "A")})
        for k in ("generation", "classement", "generations", "promos", "promo", "attente", "classement_cat", "essais"):
            etat.pop(k, None)
        etat["principales"] = {c: REFERENCE for c in CATS}
        etat.pop("dernier_recalcul", None)
        etat["journal"].insert(0, {"date": aujourdhui, "msg":
            "Nouveau moteur : une recette par catégorie (tirs 1+, 2+, 3+, 4+, but, passe, victoire, totaux), "
            "recalculée chaque semaine avec toute la saison. Départ : v1 partout."})
        etat["moteur"] = 5
        nouvelle = True

    principales = etat.get("principales") or {c: REFERENCE for c in CATS}
    for c in CATS:
        if principales.get(c) not in versions_actives and principales.get(c) not in [v["id"] for v in etat["versions"] if not v.get("retiree")]:
            principales[c] = REFERENCE
    J, M = donnees(historique)
    etat["patterns"] = patterns(J, M, principales.get("tirs2", REFERENCE))
    etat["donnees"] = {"joueurs": len(J), "matchs": len(M)}

    derniere = etat.get("dernier_recalcul")
    jours = (date.fromisoformat(aujourdhui) - date.fromisoformat(derniere)).days if derniere else 999
    if force or jours >= JOURS_RECALCUL:
        rapports = []
        for cat in CATS:
            numeros = [int(v["id"].split("_v")[1]) for v in etat["versions"]
                       if v.get("cat") == cat and "_v" in v["id"] and v["id"].split("_v")[1].isdigit()]
            numero = max(numeros, default=PREMIERE_VERSION - 1) + 1
            try:
                version, rapport = recalculer(cat, J, M, principales[cat], aujourdhui, numero)
            except Exception as ex:  # noqa: BLE001  (une catégorie qui plante n'arrête pas les autres)
                version, rapport = None, {"cat": cat, "msg": f"erreur : {str(ex)[:100]}"}
            rapports.append(rapport)
            if version:
                ancien = principales[cat]
                for v in etat["versions"]:
                    if v["id"] == ancien:
                        v["retiree"] = True       # l'ancienne recette de la catégorie arrête
                etat["versions"].append(version)
                principales[cat] = version["id"]
                nouvelle = True
            etat["journal"].insert(0, {"date": aujourdhui, "msg": f"{NOMS_CAT[cat]} : {rapport['msg']}"})
        etat["dernier_recalcul"] = aujourdhui
        etat["recalcul"] = {"date": aujourdhui, "rapports": rapports}

    etat["principales"] = principales
    etat["principale"] = principales.get("tirs2", REFERENCE)
    etat["prochain_recalcul"] = (date.fromisoformat(etat["dernier_recalcul"]) + timedelta(days=JOURS_RECALCUL)).isoformat() \
        if etat.get("dernier_recalcul") else aujourdhui
    etat["journal"] = etat["journal"][:40]
    etat["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(chemin_etat, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(etat, ensure_ascii=False, indent=1))
    return etat, nouvelle
