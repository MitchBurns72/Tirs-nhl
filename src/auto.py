"""
Le labo : l'algo qui s'améliore tout seul, AVEC des garde-fous.

Trois morceaux :

1. DÉTECTEUR DE PATTERNS (à chaque calcul)
   Regarde les prédictions gradées de la version principale, coupées en segments
   (défenseurs, back-to-back, gros temps de glace, adversaire passoire...).
   Si un segment rate systématiquement (ex. « défenseurs : prédit 41 %, réel 35 % »)
   et que l'écart est trop gros pour être de la chance (z >= 2.5), il le rapporte.
   C'est le « reverse engineering » : on voit OÙ l'algo se trompe.

2. TUNER (une fois par semaine)
   Comme un autotune de PID : il essaie des variantes des paramètres de v1 sur les
   VIEUX soirs (entraînement), puis vérifie sur les soirs les PLUS RÉCENTS qu'il
   n'a jamais utilisés pour chercher (validation). POURQUOI cette séparation :
   un algo qui s'ajuste tout seul finit toujours par « trouver » des patterns dans
   le bruit (overfitting). Le seul juge honnête, c'est des données jamais vues.
   Si la variante bat la version principale en validation avec une vraie marge,
   et n'empire aucun autre marché, elle devient une NOUVELLE version gelée
   (ex. a1026) qui roule en parallèle. Aucune version existante n'est touchée.

3. PROMOTION (à chaque calcul)
   Une version auto devient la principale (celle affichée par défaut) seulement
   quand elle bat la principale actuelle EN DIRECT, sur les mêmes joueurs et les
   mêmes soirs, après assez de prédictions. Pas sur ses données d'entraînement.

Limite assumée : le tuner ajuste des poids, il n'invente pas de nouvelles infos.
Les gros gains viendraient de nouvelles données (temps de PP, gardien partant).
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
                J.append((d, e["f"], e["result"], e.get("preds", {})))
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
# 3. Évolution par générations (appelée par run.py à chaque calcul)
# =================================================================================
#
# Comme un concours entre 5 recettes qui cuisent en même temps :
#   - Génération 2 : v2.1 à v2.5, réglées à la main AVANT de voir les résultats,
#     donc rejouées honnêtement sur tout l'historique.
#   - Quand chaque membre a assez de prédictions comparables, on prend le meilleur
#     (le « parent »), il reste en vie, et on lui fait 5 enfants : v3.1 = le tuner
#     part du parent, v3.2 à v3.5 = mutations au hasard de quelques paramètres.
#     Les 4 perdants arrêtent de prédire (leur historique reste).
#   - Génération 3 et + : les enfants sont nés en regardant l'historique, donc ils
#     sont jugés SEULEMENT sur les soirs après leur naissance (sinon c'est de la
#     triche : on choisirait celui qui a le mieux « appris par cœur » le passé).
#
# Le parent est choisi même si son avance n'est pas prouvée (il faut bien explorer
# quelque part), mais il devient PRINCIPAL (celui affiché par défaut) seulement si
# son avance est statistiquement solide : z >= 2.33 (moins de 1 % de chance d'être
# de la chance, ce qui compense le fait qu'on compare 5 candidats à la fois).

JOURS_GENERATION = 7    # une génération par semaine
# Champion par catégorie : lignes comparables exigées (tirs = 4 lignes par joueur, totaux = 2 par match)
PROMO_MIN_CAT = {"tirs": 6000, "buts": 1500, "passes": 1500, "victoire": 150, "totaux": 300}
PROMO_GAIN_CAT = 0.002  # 0.2 % d'erreur en moins au minimum...
Z_PROMO_CAT = 2.5       # ...et z >= 2.5 (on compare ~8 versions dans 5 catégories : on est plus sévère)
GEN_MIN_SOIRS_DIRECT = 3  # soirs en direct (après la naissance) exigés avant de conclure une génération
PROMO_MIN_DIRECT = 1000   # joueurs en direct exigés avant qu'une version auto devienne principale
Z_MIN_SOIRS = 7         # en bas de 7 soirs, le z n'a pas de sens (2 soirs = 2 points pour estimer le bruit)
Z_PROMO = 2.33
N_ENFANTS = 5
REFERENCE = "v1"        # étalon fixe : tout le monde est mesuré contre v1 sur les mêmes lignes

# Les 6 marchés joueur : (stat, seuil)
MARCHES_L = {"tirs1": ("tirs", 1), "tirs": ("tirs", 2), "tirs3": ("tirs", 3), "tirs4": ("tirs", 4),
             "buts": ("buts", 1), "passes": ("passes", 1)}


def _pertes_match(pv, h, a):
    """Brier victoire + total 5.5 + total 6.5 d'une prédiction de match, par marché."""
    po = pv.get("p_over") or {}
    out = {"victoire": (pv["p_home"] - (1 if h > a else 0)) ** 2}
    for ligne in ("5.5", "6.5"):
        if ligne in po:
            out[ligne] = (po[ligne] - (1 if h + a > float(ligne) else 0)) ** 2
    return out


# Les catégories : chacune a son champion (voir cycle). Les 4 seuils de tirs viennent de la
# même courbe, ils forment une seule catégorie ; les deux totaux aussi.
CATEGORIES = {"tirs": ("tirs1", "tirs", "tirs3", "tirs4"), "buts": ("buts",), "passes": ("passes",),
              "victoire": ("victoire",), "totaux": ("5.5", "6.5")}
CAT_JOUEUR = ("tirs", "buts", "passes")
NOMS_CAT = {"tirs": "Tirs (1+ à 4+)", "buts": "But", "passes": "Passe", "victoire": "Victoire", "totaux": "Totaux 5.5 / 6.5"}


def _z_par_soir(par_soir):
    """z « robuste par soir ». POURQUOI pas un z ligne par ligne : les 400 joueurs d'un
    même soir ne sont pas indépendants (même match, mêmes gardiens, même rythme).
    On traite chaque soir comme un bloc : c'est plus sévère, donc plus honnête."""
    D = sum(d for d, _ in par_soir.values())
    N = sum(n for _, n in par_soir.values())
    if len(par_soir) < Z_MIN_SOIRS or not N:
        return None
    mu = D / N
    var = sum((d - n * mu) ** 2 for d, n in par_soir.values())
    return D / math.sqrt(var) if var > 0 else None


def comparer(J, M, ref, cand, depuis=None):
    """`cand` contre `ref` sur EXACTEMENT les mêmes prédictions gradées.
    gain > 0 = cand se trompe moins (Brier plus bas). Joueurs et matchs séparés."""
    sa = sb = 0.0
    n = n_tirs = 0
    soirs = {}
    for d, _, r, preds in J:
        if depuis and d < depuis:
            continue
        pa, pb = preds.get(ref), preds.get(cand)
        if not pa or not pb:
            continue
        for m, (stat, seuil) in MARCHES_L.items():
            if stat in r and m in pa and m in pb:
                y = 1 if r[stat] >= seuil else 0
                la, lb = (pa[m]["p"] - y) ** 2, (pb[m]["p"] - y) ** 2
                sa += la; sb += lb; n += 1
                x = soirs.setdefault(d, [0.0, 0]); x[0] += la - lb; x[1] += 1
        n_tirs += "tirs" in r and "tirs" in pa and "tirs" in pb
    ma = mb = 0.0
    nm = 0
    soirs_m = {}
    for d, _, (h, a), preds in M:
        if depuis and d < depuis:
            continue
        pa, pb = preds.get(ref), preds.get(cand)
        if not pa or not pb:
            continue
        qa, qb = _pertes_match(pa, h, a), _pertes_match(pb, h, a)
        for k in qa.keys() & qb.keys():
            la, lb = qa[k], qb[k]
            ma += la; mb += lb; nm += 1
            x = soirs_m.setdefault(d, [0.0, 0]); x[0] += la - lb; x[1] += 1
    return {"n": n, "n_tirs": n_tirs, "soirs": len(soirs),
            "gain": (sa - sb) / sa if sa else None, "z": _z_par_soir(soirs),
            "brier": sb / n if n else None,
            "n_m": nm, "gain_m": (ma - mb) / ma if ma else None, "z_m": _z_par_soir(soirs_m)}


def comparer_cat(J, M, ref, cand, cat, depuis=None):
    """Comme comparer(), mais sur UNE catégorie (tirs, buts, passes, victoire, totaux)."""
    sa = sb = 0.0
    n = 0
    soirs = {}
    marches = CATEGORIES[cat]
    if cat in CAT_JOUEUR:
        for d, _, r, preds in J:
            if depuis and d < depuis:
                continue
            pa, pb = preds.get(ref), preds.get(cand)
            if not pa or not pb:
                continue
            for m in marches:
                stat, seuil = MARCHES_L[m]
                if stat in r and m in pa and m in pb:
                    y = 1 if r[stat] >= seuil else 0
                    la, lb = (pa[m]["p"] - y) ** 2, (pb[m]["p"] - y) ** 2
                    sa += la; sb += lb; n += 1
                    x = soirs.setdefault(d, [0.0, 0]); x[0] += la - lb; x[1] += 1
    else:
        for d, _, (h, a), preds in M:
            if depuis and d < depuis:
                continue
            pa, pb = preds.get(ref), preds.get(cand)
            if not pa or not pb:
                continue
            qa, qb = _pertes_match(pa, h, a), _pertes_match(pb, h, a)
            for k in marches:
                if k in qa and k in qb:
                    sa += qa[k]; sb += qb[k]; n += 1
                    x = soirs.setdefault(d, [0.0, 0]); x[0] += qa[k] - qb[k]; x[1] += 1
    return {"n": n, "soirs": len(soirs), "gain": (sa - sb) / sa if sa else None, "z": _z_par_soir(soirs),
            "brier": sb / n if n else None}


def _cle_match(k):
    """Les paramètres du modèle de MATCH (indépendants de ceux des joueurs)."""
    # cal_a / cal_b = calibration de la victoire ; cal_t_*, cal_bu_*, cal_pa_* sont aux JOUEURS
    return k.startswith(("m_", "mt_")) or k in ("cal_a", "cal_b", "w_mod_match", "scale_match")


def _espace(fam):
    joueurs = [e for g in ("tirs", "buts", "passes") for e in ESPACE[g] + (ESPACE_V2[g] if fam == "v2" else [])]
    match = ESPACE["match"] + (ESPACE_V2["match"] if fam == "v2" else [])
    return joueurs, match


def muter(P, fam, rng, n_joueur=2, n_match=1, force=0.4, radical=False):
    """Change quelques paramètres au hasard, dans des bornes raisonnables.
    radical=True : tire la valeur n'importe où entre les bornes (pas juste autour de
    la valeur actuelle). C'est le « saut » qui peut trouver une autre recette au lieu
    de raffiner la même."""
    P = dict(P)
    joueurs, match = _espace(fam)
    changes = []
    if radical:
        # Le saut vise surtout les TIRS (le but premier de la patente) : 2/3 des paramètres joueurs
        tirs = ESPACE["tirs"] + (ESPACE_V2["tirs"] if fam == "v2" else [])
        reste = [e for e in joueurs if e not in tirs]
        n_t = min(len(tirs), max(1, round(n_joueur * 2 / 3)))
        choix = rng.sample(tirs, n_t) + rng.sample(reste, min(len(reste), n_joueur - n_t))
    else:
        choix = rng.sample(joueurs, min(n_joueur, len(joueurs)))
    for nom, lo, hi, pas in choix + rng.sample(match, min(n_match, len(match))):
        val = P.get(nom, 1.0)
        if radical:
            # log-uniforme pour les paramètres « multiplicatifs » strictement positifs
            neuf = math.exp(rng.uniform(math.log(lo), math.log(hi))) if pas == "mult" and lo > 0 else rng.uniform(lo, hi)
        elif pas == "mult":
            neuf = val * math.exp(rng.uniform(-force, force)) if val else lo + (hi - lo) * rng.uniform(0.05, 0.3) * (1 + force)
        else:
            neuf = val + rng.uniform(-0.1, 0.1) * force
        neuf = round(min(max(neuf, lo), hi), 4)
        if neuf != val:
            P[nom] = neuf
            changes.append(f"{nom} {val:.3g}→{neuf:.3g}")
    return P, changes


def _params_de(vid, etat):
    """Paramètres complets d'une version (manuelle ou auto)."""
    if vid in VERSIONS and "params" in VERSIONS[vid]:
        return dict(VERSIONS[vid]["params"]), VERSIONS[vid].get("famille", "v1")
    v = next((x for x in etat["versions"] if x["id"] == vid), None)
    fam = (v or {}).get("famille", "v1")
    return {**FAMILLES[fam]["P"], **((v or {}).get("params") or {})}, fam


def _ecart(Pa, Pb, echantillon):
    """Différence moyenne des % de tirs 2+ entre deux jeux de paramètres. POURQUOI :
    deux enfants aux paramètres différents peuvent donner presque les mêmes % (on l'a
    vu en génération 3 : tout à 0.1 % près). Tester 5 fois la même recette = gaspillage."""
    if not echantillon:
        return 1.0
    s = 0.0
    n = 0
    for _, f, _, _ in echantillon:
        a, b = _FAM["joueur"](f, Pa), _FAM["joueur"](f, Pb)
        # % du modèle SEUL (p0) : le mélange avec le marché écrase les différences de recette
        for m in ("tirs", "buts", "passes"):
            s += abs(a[m].get("p0", a[m]["p"]) - b[m].get("p0", b[m]["p"])); n += 1
    return s / n


def calibrer(J, P):
    """Corrige les biais d'ensemble (le modèle voit trop haut ou trop bas) : descente
    seulement sur l'échelle, la dispersion et la recalibration, entraînée sur les vieux
    soirs, gardée seulement si ça tient sur les soirs récents."""
    J = sorted(J, key=lambda r: r[0])[-40000:]
    dates = sorted({r[0] for r in J})
    if len(dates) < 4:
        return P, []
    coupe = dates[int(len(dates) * 0.7)]
    global DERNIERE_COUPE
    DERNIERE_COUPE = coupe
    Jt, Jv = [r for r in J if r[0] < coupe], [r for r in J if r[0] >= coupe]
    P = _tuple_caps(P)
    notes = []
    cibles = {"tirs": [("scale_tirs", 0.85, 1.15, "add"), ("nb_k", 3, 200, "mult"),
                       ("cal_t_a", 0.7, 1.3, "add"), ("cal_t_b", -0.3, 0.3, "add")],
              "buts": [("scale_buts", 0.80, 1.20, "add"), ("cal_bu_a", 0.7, 1.3, "add"), ("cal_bu_b", -0.3, 0.3, "add")],
              "passes": [("scale_passes", 0.80, 1.20, "add"), ("cal_pa_a", 0.7, 1.3, "add"), ("cal_pa_b", -0.3, 0.3, "add")]}
    for groupe, esp in cibles.items():
        cand = _descente(groupe, Jt, P, passes=3, espace=esp)
        avant, apres = _perte(groupe, Jv, P), _perte(groupe, Jv, cand)
        if avant and apres is not None and apres < avant:
            P = cand
            notes.append(f"{NOMS[groupe]} {(avant - apres) / avant * 100:.1f} % mieux en validation")
    return P, notes


# Les ajouts propres à la v2, côté joueurs (ce qui la distingue de v1)
AJOUTS_V2 = ("w_att", "k_pp", "pen_exp", "score_s", "k_pp_passes")


def _juge_de(vid, etat):
    v = next((x for x in etat["versions"] if x["id"] == vid), None) or VERSIONS.get(vid) or {}
    return v.get("juge_depuis")


def _nouvelle_generation(etat, J, M, classement, aujourdhui):
    """Le meilleur survit et a 5 enfants aux « personnalités » différentes.
    Retourne le message du journal."""
    global _FAM
    gen = etat["generation"]
    n = gen["n"] + 1
    best_j = classement[0]["id"]
    # Recette combinée : chaque groupe de paramètres vient du meilleur membre DANS SA
    # catégorie (tirs, but, passe, matchs). Les paramètres de base (forme, temps de glace...)
    # viennent du meilleur aux tirs, le but premier.
    def meilleur_cat(cats):
        notes = {}
        for c in classement:
            if (c.get("n_tirs") or 0) < 500:
                continue
            notes[c["id"]] = sum((comparer_cat(J, M, REFERENCE, c["id"], cat, _juge_de(c["id"], etat))["gain"] or 0)
                                 for cat in cats)
        return max(notes, key=notes.get) if notes else best_j
    best_t, best_b, best_p = meilleur_cat(["tirs"]), meilleur_cat(["buts"]), meilleur_cat(["passes"])
    best_m = meilleur_cat(["victoire", "totaux"])
    P, fam = _params_de(best_t, etat)
    cles = {"buts": {e[0] for e in ESPACE["buts"] + ESPACE_V2["buts"]},
            "passes": {e[0] for e in ESPACE["passes"] + ESPACE_V2["passes"]}}
    for source, garde in ((best_b, lambda k: k in cles["buts"]), (best_p, lambda k: k in cles["passes"]),
                          (best_m, _cle_match)):
        Pk, famk = _params_de(source, etat)
        if famk == fam:
            P.update({k: v for k, v in Pk.items() if garde(k)})
    recette = f"tirs de {best_t}, but de {best_b}, passe de {best_p}, matchs de {best_m}"
    _FAM = FAMILLES[fam]
    base = FAMILLES[fam]["P"]
    rng = random.Random(f"{aujourdhui}-gen{n}")   # même date = mêmes mutations (rejouable)
    echantillon = [r for r in sorted(J, key=lambda r: r[0])[-600:] if r[1].get("x")][::2]
    perd_contre_v1 = (classement[0]["gain"] or 0) < 0
    dejavu = [P]

    def distinct(fabrique, minimum, essais=5):
        """Refait la mutation, plus fort à chaque fois, jusqu'à ce que l'enfant soit
        vraiment différent du parent ET des frères déjà nés."""
        Pk, ch = fabrique(1.0)
        for i in range(essais):
            if min(_ecart(Pk, autre, echantillon) for autre in dejavu) >= minimum:
                break
            Pk, ch = fabrique(1.0 + 0.6 * (i + 1))
        dejavu.append(Pk)
        return Pk, ch

    enfants = []
    # 1. LE TUNER : raffine le parent (cherche sur le vieux, valide sur le récent)
    Pt, msg_t = tuner(J, M, P, aujourdhui) if len(J) >= MIN_JOUEURS or len(M) >= MIN_MATCHS else (None, "pas assez de données")
    juge = {}   # index de l'enfant -> premier soir qui compte pour le juger
    if Pt:
        dejavu.append(Pt)
        juge[0] = DERNIERE_COUPE
        enfants.append((Pt, f"Tuner : raffine la recette combinée ({recette}). {msg_t}"))
    else:
        Pk, ch = distinct(lambda k: muter(P, fam, rng, 2, 1, 0.3 * k), 0.006)
        enfants.append((Pk, f"Tuner : rien trouvé, petite mutation de la recette à la place : " + ", ".join(ch)))

    # 2. LE CALIBREUR : corrige les biais d'ensemble (ex. « voit trop de tirs »)
    Pc, notes = calibrer(J, P)
    if notes:
        dejavu.append(Pc)
        juge[len(enfants)] = DERNIERE_COUPE
        enfants.append((Pc, f"Calibreur : recette combinée recalibrée (échelle, dispersion, biais). " + " · ".join(notes)))
    else:
        Pk, ch = distinct(lambda k: muter(P, fam, rng, 2, 0, 0.4 * k), 0.008)
        enfants.append((Pk, f"Calibreur : aucun biais à corriger, mutation de la recette à la place : " + ", ".join(ch)))

    # 3. L'EXPÉRIENCE : si toute la génération perd contre v1, on teste si ce sont les
    #    ajouts de la v2 qui nuisent (on les coupe à 40 %). Sinon : croisement des 2 meilleurs.
    if perd_contre_v1 and fam == "v2":
        Pe = dict(P)
        ch = []
        for k in AJOUTS_V2:
            if Pe.get(k):
                Pe[k] = round(Pe[k] * 0.4, 4)
                ch.append(f"{k} {P[k]:.3g}→{Pe[k]:.3g}")
        dejavu.append(Pe)
        enfants.append((Pe, f"Expérience : {best_j} perd contre v1, alors on coupe les ajouts v2 à 40 % "
                            f"pour voir si c'est eux qui nuisent : " + ", ".join(ch)))
    else:
        # Croisement avec le 2e, seulement s'il est vraiment différent (sinon c'est un clone)
        Px, ch, autre = None, [], classement[1]["id"] if len(classement) > 1 else None
        if autre:
            P2, fam2 = _params_de(autre, etat)
            diff = sorted(k for k in P2 if fam2 == fam and P.get(k) != P2[k])
            if diff:
                pris = [k for k in diff if rng.random() < 0.5] or [rng.choice(diff)]
                Px = dict(P, **{k: P2[k] for k in pris})
                ch = pris
        if Px is not None and _ecart(Px, P, echantillon) >= 0.004:
            dejavu.append(Px)
            enfants.append((Px, f"Croisement : {best_j} × {autre} (prend de {autre} : {', '.join(ch)})"))
        else:
            Pk, ch = distinct(lambda k: muter(P, fam, rng, 3, 1, 0.5 * k), 0.008)
            enfants.append((Pk, f"Mutation moyenne de la recette (le 2e est presque un clone, croisement inutile) : " + ", ".join(ch)))

    # 4. L'EXPLORATRICE : grosse mutation (5 paramètres joueurs + 2 match)
    Pk, ch = distinct(lambda k: muter(P, fam, rng, 5, 2, 0.8 * k), 0.012)
    enfants.append((Pk, f"Exploratrice : grosse mutation de la recette : " + ", ".join(ch)))

    # 5. LA RADICALE : 7 paramètres tirés n'importe où dans leurs bornes
    Pk, ch = distinct(lambda k: muter(P, fam, rng, 6, 2, 1.0, radical=True), 0.02)
    enfants.append((Pk, f"Radicale : saut loin de la recette : " + ", ".join(ch)))

    ids = []
    for i, (Pk, desc) in enumerate(enfants[:N_ENFANTS], start=1):
        vid = f"v{n}.{i}"
        diff = {k: (list(v) if isinstance(v, tuple) else v) for k, v in Pk.items()
                if (list(base[k]) if isinstance(base.get(k), tuple) else base.get(k)) != (list(v) if isinstance(v, tuple) else v)}
        etat["versions"] = [v for v in etat["versions"] if v["id"] != vid]
        etat["versions"].append({"id": vid, "cree": aujourdhui, "base": best_j, "famille": fam, "generation": n,
                                 "params": diff, "desc": desc, "retiree": False,
                                 "ecart": round(_ecart(Pk, P, echantillon), 4),
                                 # None = rejouée et jugée sur toute la saison (paramètres pas ajustés
                                 # sur les données) ; une date = seulement à partir de ses soirs de validation
                                 "juge_depuis": juge.get(i - 1)})
        ids.append(vid)

    # Trop jeunes pour être jugés (moins de 500 joueurs notés, ex. une VA ajoutée en cours de
    # semaine) : ils passent à la génération suivante au lieu d'être éliminés sans procès.
    jeunes = [c["id"] for c in classement if (c.get("n_tirs") or 0) < 500 and c["id"] != best_j]
    # Les perdants arrêtent de prédire. Jamais la principale, jamais le parent.
    champions = set((etat.get("principales") or {}).values()) | {etat.get("principale")}
    perdants = [c["id"] for c in classement if c["id"] != best_j and c["id"] not in champions
                and not c.get("hors_gen") and c["id"] not in jeunes]
    for vid in perdants:
        v = next((x for x in etat["versions"] if x["id"] == vid), None)
        if v:
            v["retiree"] = True
        elif vid not in etat.setdefault("retirees", []):
            etat["retirees"].append(vid)

    gagnant = classement[0]
    etat.setdefault("generations", []).insert(0, {
        "n": gen["n"], "fin": aujourdhui, "gagnant": best_j, "gain": gagnant["gain"], "z": gagnant["z"],
        "classement": [{k: c.get(k) for k in ("id", "gain", "z", "gain_m")} for c in classement]})
    etat["generation"] = {"n": n, "membres": [best_j] + ids + jeunes, "parent": best_j, "depuis": None, "debut": aujourdhui}
    return (f"Génération {gen['n']} terminée : {best_j} gagne ({(gagnant['gain'] or 0) * 100:+.2f} % vs {REFERENCE}"
            + (f", z = {gagnant['z']:.1f}" if gagnant["z"] is not None else "") + f"). Recette des enfants : {recette}. Génération {n} : "
            + ", ".join(ids) + f" naissent et affrontent {best_j}. Perdants retirés : {', '.join(perdants) or 'aucun'}.")


def cycle(historique, chemin_etat, versions_actives, aujourdhui=None, force=False):
    """Patterns + évolution (classement, promotion, nouvelle génération).
    Met à jour data/versions_auto.json. Retourne (état, True si de nouvelles versions sont nées)."""
    aujourdhui = aujourdhui or date.today().isoformat()
    try:
        etat = json.loads(open(chemin_etat, encoding="utf-8").read())
    except (FileNotFoundError, ValueError):
        etat = {"versions": [], "journal": [], "principale": REFERENCE}
    etat.setdefault("versions", []); etat.setdefault("journal", [])
    global _FAM
    principale = etat.get("principale", REFERENCE)
    if principale not in versions_actives:
        principale = REFERENCE
    J, M = donnees(historique)
    # Les patterns regardent la principale : c'est elle qu'on veut corriger.
    etat["patterns"] = patterns(J, M, principale)
    etat["donnees"] = {"joueurs": len(J), "matchs": len(M)}
    etat.pop("essais", None)   # ancien format (une seule v2)
    nouvelle = False

    # Première fois : la génération 2 = les versions manuelles marquées generation 2
    if "generation" not in etat:
        membres = [v for v, s in VERSIONS.items() if s.get("generation") == 2 and v in versions_actives]
        etat["generation"] = {"n": 2, "membres": membres, "parent": None, "depuis": None, "debut": aujourdhui}
    gen = etat["generation"]

    # --- Migration unique (23 octobre) : la génération 3 de l'ancien moteur avait 5 enfants
    # presque identiques au parent (mêmes % à 0.1 % près). On l'interrompt et on refait
    # naître la génération suivante avec le nouveau moteur (5 personnalités, enfants
    # garantis différents). Les vieilles v3.x gardent leur historique. ---
    if etat.get("moteur", 1) < 2 and gen["n"] >= 3 and gen.get("parent") and etat.get("generations"):
        parent = gen["parent"]
        anciens = [m for m in gen["membres"] if m != parent]
        etat_g = [{"id": m, **comparer(J, M, REFERENCE, m, gen.get("depuis"))} for m in gen["membres"]]
        for v in etat["versions"]:
            if v["id"] in anciens:
                v["retiree"] = True
        precedent = etat["generations"][0]
        ref = [{"id": parent, **{k: precedent.get(k) for k in ("gain", "z")},
                "gain_m": next((c.get("gain_m") for c in precedent.get("classement", []) if c["id"] == parent), None)}]
        msg = _nouvelle_generation(etat, J, M, ref, aujourdhui)
        # L'entrée d'historique créée décrit la génération interrompue, pas la précédente
        etat["generations"][0] = {"n": gen["n"], "fin": aujourdhui, "gagnant": parent, "interrompue": True,
                                  "gain": None, "z": None,
                                  "classement": [{k: c.get(k) for k in ("id", "gain", "z", "gain_m")} for c in etat_g]}
        etat["journal"].insert(0, {"date": aujourdhui, "msg":
            f"Nouveau moteur : génération {gen['n']} interrompue (ses 5 enfants étaient presque des clones de {parent}). "
            + msg.split(". ", 1)[1]})
        etat["moteur"] = 3
        nouvelle = True
        gen = etat["generation"]
    # --- Migration (moteur 3, 23 octobre) : les versions auto sont maintenant rejouées sur
    # toute la saison. Celles dont les paramètres ont été AJUSTÉS sur les données (Tuner,
    # Calibreur) comptent seulement à partir de leurs soirs de validation. ---
    if etat.get("moteur", 2) < 3:
        for v in etat["versions"]:
            if "juge_depuis" in v or v.get("retiree"):
                continue
            ajuste = v["desc"].startswith("Tuner : raffine") or (v["desc"].startswith("Calibreur") and "recalibré" in v["desc"])
            if ajuste:
                ds = sorted({r[0] for r in J if r[0] < v["cree"]})
                v["juge_depuis"] = ds[int(len(ds) * 0.7)] if len(ds) >= 4 else v["cree"]
            else:
                v["juge_depuis"] = None
        gen["depuis"] = None
        etat["moteur"] = 3
    infos = {v["id"]: v for v in etat["versions"]}
    # Versions auto (data/versions_auto.json) ou manuelles comme va.1 (modeles.py)
    juge = lambda vid: (infos.get(vid) or VERSIONS.get(vid) or {}).get("juge_depuis")
    cree = lambda vid: (infos.get(vid) or VERSIONS.get(vid) or {}).get("cree")
    # Les VA (scan complet des paramètres, fait sur demande) rejoignent le concours en cours
    for vid, spec in VERSIONS.items():
        if spec.get("generation") == "A" and vid in versions_actives and vid not in gen["membres"]:
            gen["membres"].append(vid)
            etat["journal"].insert(0, {"date": aujourdhui, "msg": f"{vid} (scan complet des paramètres) rejoint la génération {gen['n']}."})
    membres = [m for m in gen["membres"] if m in versions_actives or (m in infos and not infos[m].get("retiree"))]

    # --- Classement de la génération en cours (tous contre v1, mêmes lignes) ---
    # Chaque membre est jugé sur toute la saison (rejouée), sauf les soirs qui ont servi à
    # ajuster ses paramètres. « direct » = soirs prédits AVANT le match, après sa naissance.
    classement = []
    for m in membres:
        c = comparer(J, M, REFERENCE, m, juge(m) or gen.get("depuis"))
        d = comparer(J, M, REFERENCE, m, cree(m)) if cree(m) else None
        classement.append({"id": m, **c, "soirs_direct": d["soirs"] if d else None,
                           "gain_direct": d["gain"] if d else None, "juge_depuis": juge(m)})
    classement.sort(key=lambda c: -(c["gain"] if c["gain"] is not None else -9))
    # La principale, si elle n'est pas dans la génération (champion en titre), est
    # affichée avec le classement pour comparer, mais ne compte pas dans le concours.
    hors = []
    if principale not in membres and principale != REFERENCE:
        hors = [{"id": principale, "hors_gen": True, **comparer(J, M, REFERENCE, principale, juge(principale))}]
    etat["classement"] = classement + hors

    # --- Un champion PAR CATÉGORIE (tirs, but, passe, victoire, totaux) ---
    # La meilleure recette pour les tirs n'est pas forcément la meilleure pour les passes ou
    # les matchs. Chaque catégorie a son champion ; la version « combo » de la page prend le
    # % de chaque marché chez le champion de sa catégorie. Tout le monde est candidat (pas
    # juste la génération en cours), et un champion change seulement si c'est PROUVÉ.
    principales = etat.get("principales") or {c: principale for c in CATEGORIES}
    for c in CATEGORIES:
        if principales.get(c) not in versions_actives:
            principales[c] = REFERENCE
    candidats = [v for v in versions_actives if v != "v0"]
    etat["classement_cat"], etat["promos"] = {}, {}
    for cat in CATEGORIES:
        lignes = []
        for v in candidats:
            r = comparer_cat(J, M, REFERENCE, v, cat, juge(v))
            if r["n"]:
                lignes.append({"id": v, **{k: r[k] for k in ("n", "soirs", "gain", "z")}})
        lignes.sort(key=lambda x: -(x["gain"] or 0))
        etat["classement_cat"][cat] = lignes
        champ = principales[cat]
        mini = PROMO_MIN_CAT[cat]
        meilleur = next((x for x in lignes if x["n"] >= mini), None)
        if not meilleur or meilleur["id"] == champ:
            continue
        b = meilleur["id"]
        c = comparer_cat(J, M, champ, b, cat, max(filter(None, [juge(b), juge(champ)]), default=None))
        solide = (c["n"] >= mini and (c["gain"] or 0) >= PROMO_GAIN_CAT and c["z"] is not None and c["z"] >= Z_PROMO_CAT)
        direct = None
        if cree(b):
            # Une version auto ou VA doit AUSSI tenir en direct (après sa naissance)
            direct = comparer_cat(J, M, champ, b, cat, cree(b))
            solide = solide and direct["n"] >= mini // 3 and (direct["gain"] or 0) > 0
        etat["promos"][cat] = {"candidat": b, "contre": champ, "n": c["n"], "gain": c["gain"], "z": c["z"],
                               "solide": solide, "direct_n": direct["n"] if direct else None,
                               "direct_gain": direct["gain"] if direct else None}
        if solide:
            etat["journal"].insert(0, {"date": aujourdhui, "msg":
                f"{NOMS_CAT[cat]} : {b} devient champion ({c['gain'] * 100:+.2f} % vs {champ}, z = {c['z']:.1f})."})
            principales[cat] = b
    etat["principales"] = principales
    # Compatibilité : « principale » = le champion des tirs (le but premier de la patente)
    principale = principales["tirs"]
    etat["principale"] = principale
    etat.pop("promo", None)

    # --- Fin de génération : chacun a assez de données comparables, ET quelques soirs en
    # direct (sinon on choisirait toujours sur les mêmes vieux soirs, encore et encore) ---
    # Une génération par semaine : on analyse tout le monde avec la semaine de nouvelles
    # données, le meilleur a 5 enfants. Pas 12 000 versions d'un coup : le rythme suit
    # l'arrivée de données fraîches.
    jours_gen = (date.fromisoformat(aujourdhui) - date.fromisoformat(gen["debut"])).days if gen.get("debut") else 99
    def pret_membre(c):
        return c["soirs_direct"] is None or c["soirs_direct"] >= GEN_MIN_SOIRS_DIRECT
    pret = classement and jours_gen >= JOURS_GENERATION and all(pret_membre(c) for c in classement)
    if classement and (pret or force):
        etat["principale"] = principale
        msg = _nouvelle_generation(etat, J, M, classement, aujourdhui)
        etat["journal"].insert(0, {"date": aujourdhui, "msg": msg})
        etat["dernier_tuning"] = aujourdhui
        etat.pop("attente", None)
        nouvelle = True
    elif classement:
        manque = min(classement, key=lambda c: c["n_tirs"])
        directs = [c["soirs_direct"] for c in classement if c["soirs_direct"] is not None]
        etat["attente"] = {"n_tirs": manque["n_tirs"], "soirs": manque["soirs"],
                           "fin": (date.fromisoformat(gen["debut"]) + timedelta(days=JOURS_GENERATION)).isoformat(),
                           "soirs_direct": min(directs) if directs else None, "min_direct": GEN_MIN_SOIRS_DIRECT}

    etat["journal"] = etat["journal"][:30]
    etat["principale"] = principale
    etat["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(chemin_etat, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(etat, ensure_ascii=False, indent=1))
    return etat, nouvelle
