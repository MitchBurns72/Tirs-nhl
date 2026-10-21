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
from datetime import date, datetime, timezone

from modeles import FAMILLES, V1, VERSIONS

# Famille de modèle en cours de tuning (v1 ou v2) : fixée par cycle() selon la principale
_FAM = FAMILLES["v1"]

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
             ("score_s", 0.0, 0.4, "mult"), ("w_mod_joueur", 0.1, 1.0, "mult"), ("vig_props", 1.0, 1.15, "add")],
    "buts": [],
    "passes": [("k_pp_passes", 0.0, 4.0, "mult")],
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


def _descente(groupe, rows, P, passes=2):
    """Descente par coordonnées : un paramètre à la fois, on garde ce qui baisse la perte."""
    P = dict(P)
    meilleur = _perte(groupe, rows, P)
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

GEN_MIN_TIRS = 3000     # lignes de joueurs comparables pour CHAQUE membre avant de conclure
GEN_MIN_SOIRS = 7       # et au moins 7 soirs différents (un soir bizarre ne décide rien)
Z_PROMO = 2.33
N_ENFANTS = 5
REFERENCE = "v1"        # étalon fixe : tout le monde est mesuré contre v1 sur les mêmes lignes

# Les 6 marchés joueur : (stat, seuil)
MARCHES_L = {"tirs1": ("tirs", 1), "tirs": ("tirs", 2), "tirs3": ("tirs", 3), "tirs4": ("tirs", 4),
             "buts": ("buts", 1), "passes": ("passes", 1)}


def _pertes_match(pv, h, a):
    """Brier victoire + total 5.5 + total 6.5 d'une prédiction de match."""
    po = pv.get("p_over") or {}
    out = [(pv["p_home"] - (1 if h > a else 0)) ** 2]
    for ligne in ("5.5", "6.5"):
        if ligne in po:
            out.append((po[ligne] - (1 if h + a > float(ligne) else 0)) ** 2)
    return out


def _z_par_soir(par_soir):
    """z « robuste par soir ». POURQUOI pas un z ligne par ligne : les 400 joueurs d'un
    même soir ne sont pas indépendants (même match, mêmes gardiens, même rythme).
    On traite chaque soir comme un bloc : c'est plus sévère, donc plus honnête."""
    D = sum(d for d, _ in par_soir.values())
    N = sum(n for _, n in par_soir.values())
    if len(par_soir) < 2 or not N:
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
        for la, lb in zip(_pertes_match(pa, h, a), _pertes_match(pb, h, a)):
            ma += la; mb += lb; nm += 1
            x = soirs_m.setdefault(d, [0.0, 0]); x[0] += la - lb; x[1] += 1
    return {"n": n, "n_tirs": n_tirs, "soirs": len(soirs),
            "gain": (sa - sb) / sa if sa else None, "z": _z_par_soir(soirs),
            "brier": sb / n if n else None,
            "n_m": nm, "gain_m": (ma - mb) / ma if ma else None, "z_m": _z_par_soir(soirs_m)}


def _cle_match(k):
    """Les paramètres du modèle de MATCH (indépendants de ceux des joueurs)."""
    return k.startswith(("m_", "mt_", "cal_")) or k in ("w_mod_match", "scale_match")


def _espace(fam):
    joueurs = [e for g in ("tirs", "buts", "passes") for e in ESPACE[g] + (ESPACE_V2[g] if fam == "v2" else [])]
    match = ESPACE["match"] + (ESPACE_V2["match"] if fam == "v2" else [])
    return joueurs, match


def muter(P, fam, rng, n_joueur=2, n_match=1, force=0.4):
    """Change quelques paramètres au hasard, dans des bornes raisonnables.
    POURQUOI peu de paramètres à la fois : si l'enfant gagne, on sait à peu près
    pourquoi. Si on brasse tout, on apprend rien même quand ça marche."""
    P = dict(P)
    joueurs, match = _espace(fam)
    changes = []
    for nom, lo, hi, pas in rng.sample(joueurs, n_joueur) + rng.sample(match, n_match):
        val = P.get(nom, 1.0)
        if pas == "mult":
            neuf = val * math.exp(rng.uniform(-force, force)) if val else lo + (hi - lo) * rng.uniform(0.05, 0.3)
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


def _nouvelle_generation(etat, J, M, classement, aujourdhui):
    """Le meilleur survit et a 5 enfants. Retourne le message du journal."""
    global _FAM
    gen = etat["generation"]
    n = gen["n"] + 1
    best_j = classement[0]["id"]
    # Croisement : les paramètres JOUEURS du meilleur sur les joueurs, et les paramètres
    # MATCH du meilleur sur les matchs (les deux modèles sont indépendants).
    best_m = max((c for c in classement if c.get("gain_m") is not None),
                 key=lambda c: c["gain_m"], default=classement[0])["id"]
    P, fam = _params_de(best_j, etat)
    Pm, fam_m = _params_de(best_m, etat)
    if fam_m == fam:
        P.update({k: v for k, v in Pm.items() if _cle_match(k)})
    _FAM = FAMILLES[fam]
    base = FAMILLES[fam]["P"]
    rng = random.Random(f"{aujourdhui}-gen{n}")   # même date = mêmes mutations (rejouable)

    enfants = []
    # Enfant 1 : le tuner part du croisement (il cherche sur le vieux, valide sur le récent)
    Pt, msg_t = tuner(J, M, P, aujourdhui) if len(J) >= MIN_JOUEURS or len(M) >= MIN_MATCHS else (None, "pas assez de données")
    if Pt:
        enfants.append((Pt, f"Tuner à partir de {best_j}" + (f" (matchs de {best_m})" if best_m != best_j else "") + f" : {msg_t}"))
    else:
        Pm2, ch = muter(P, fam, rng, 1, 1, 0.2)
        enfants.append((Pm2, f"Petite mutation de {best_j} (le tuner n'a rien trouvé) : " + ", ".join(ch)))
    # Enfants 2 à 5 : mutations au hasard, de plus en plus audacieuses
    for i, (nj, nm_, force) in enumerate([(2, 1, 0.3), (2, 1, 0.4), (3, 1, 0.5), (3, 2, 0.7)], start=2):
        Pk, ch = muter(P, fam, rng, nj, nm_, force)
        enfants.append((Pk, f"Mutation de {best_j} : " + ", ".join(ch)))

    ids = []
    for i, (Pk, desc) in enumerate(enfants[:N_ENFANTS], start=1):
        vid = f"v{n}.{i}"
        diff = {k: (list(v) if isinstance(v, tuple) else v) for k, v in Pk.items()
                if (list(base[k]) if isinstance(base.get(k), tuple) else base.get(k)) != (list(v) if isinstance(v, tuple) else v)}
        etat["versions"] = [v for v in etat["versions"] if v["id"] != vid]
        etat["versions"].append({"id": vid, "cree": aujourdhui, "base": best_j, "famille": fam, "generation": n,
                                 "params": diff, "desc": desc, "retiree": False})
        ids.append(vid)

    # Les perdants arrêtent de prédire. Jamais la principale, jamais le parent.
    perdants = [c["id"] for c in classement if c["id"] not in (best_j, etat.get("principale"))]
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
    etat["generation"] = {"n": n, "membres": [best_j] + ids, "parent": best_j, "depuis": aujourdhui, "debut": aujourdhui}
    return (f"Génération {gen['n']} terminée : {best_j} gagne ({(gagnant['gain'] or 0) * 100:+.2f} % vs {REFERENCE}"
            + (f", z = {gagnant['z']:.1f}" if gagnant["z"] is not None else "") + f"). Génération {n} : "
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
    membres = [m for m in gen["membres"] if m in versions_actives]

    # --- Classement de la génération en cours (tous contre v1, mêmes lignes) ---
    classement = []
    for m in membres:
        c = comparer(J, M, REFERENCE, m, gen.get("depuis"))
        classement.append({"id": m, **c})
    classement.sort(key=lambda c: -(c["gain"] if c["gain"] is not None else -9))
    # La principale, si elle n'est pas dans la génération (champion en titre), est
    # affichée avec le classement pour comparer, mais ne compte pas dans le concours.
    hors = []
    if principale not in membres and principale != REFERENCE:
        hors = [{"id": principale, "hors_gen": True, **comparer(J, M, REFERENCE, principale, gen.get("depuis"))}]
    etat["classement"] = classement + hors

    # --- Promotion : le meilleur devient principal seulement si c'est prouvé ---
    if classement and classement[0]["id"] != principale:
        b = classement[0]["id"]
        depuis = gen.get("depuis")
        c = comparer(J, M, principale, b, depuis)
        solide = (c["n_tirs"] >= PROMO_MIN_TIRS and c["gain"] is not None and c["gain"] >= PROMO_GAIN
                  and c["z"] is not None and c["z"] >= Z_PROMO
                  and not (c["z_m"] is not None and c["z_m"] <= -Z_PROMO))   # pas pire sur les matchs
        etat["promo"] = {"candidat": b, "contre": principale, **{k: c[k] for k in ("n_tirs", "gain", "z", "gain_m", "z_m")},
                         "solide": solide}
        if solide:
            etat["journal"].insert(0, {"date": aujourdhui, "msg":
                f"{b} devient la version principale : {c['gain'] * 100:.2f} % mieux que {principale} "
                f"sur les mêmes {c['n_tirs']} joueurs (z = {c['z']:.1f})."})
            principale = b
            etat["principale"] = b
    else:
        etat.pop("promo", None)

    # --- Fin de génération : chacun a assez de données comparables ---
    pret = classement and all(c["n_tirs"] >= GEN_MIN_TIRS and c["soirs"] >= GEN_MIN_SOIRS for c in classement)
    if pret and (force or gen.get("debut") != aujourdhui):
        etat["principale"] = principale
        msg = _nouvelle_generation(etat, J, M, classement, aujourdhui)
        etat["journal"].insert(0, {"date": aujourdhui, "msg": msg})
        etat["dernier_tuning"] = aujourdhui
        nouvelle = True
    elif classement:
        manque = min(classement, key=lambda c: c["n_tirs"])
        etat["attente"] = {"n_tirs": manque["n_tirs"], "soirs": manque["soirs"],
                           "min_tirs": GEN_MIN_TIRS, "min_soirs": GEN_MIN_SOIRS}
    if nouvelle:
        etat.pop("attente", None)

    etat["journal"] = etat["journal"][:30]
    etat["principale"] = principale
    etat["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(chemin_etat, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(etat, ensure_ascii=False, indent=1))
    return etat, nouvelle
