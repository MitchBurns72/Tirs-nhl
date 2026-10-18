"""
Registre des versions de l'algorithme.

RÈGLE D'OR : une version publiée ne se modifie JAMAIS.
On veut comparer v0 vs v1 vs v2 sur les mêmes matchs, en tout temps. Si on
retouche v1 après coup, ses vieilles prédictions ne correspondent plus au code
et la comparaison devient de la bouillie. Pour essayer une idée : on copie la
dernière version, on lui donne un nouveau nom (ex. "v2") et on l'ajoute à VERSIONS.
(Ajouter un NOUVEAU marché à une version, comme on l'a fait pour buts/passes/matchs,
c'est correct : ça ne change rien aux prédictions déjà faites sur les autres marchés.)

Toutes les versions roulent chaque jour sur la même soirée ("shadow mode").

Chaque version a deux fonctions :
  - match(ctx_match)  -> {"p_home", "lam_home", "lam_away", "total", "p_over": {"5.5", "6.5"}}
  - joueur(ctx, jeu)  -> {"tirs": {...}, "buts": {...}, "passes": {...}}
`jeu` = la prédiction de match de la MÊME version pour l'équipe du joueur
(buts attendus pour/contre). Les passes en dépendent : pas de but, pas de passe.
"""

import math


# =================================================================================
# Distributions
# =================================================================================

def p2plus_poisson(lam):
    """P(X >= 2) pour une Poisson : 1 - P(0) - P(1) = 1 - e^-λ(1 + λ)."""
    return 1.0 - math.exp(-lam) * (1.0 + lam)


def p1plus_poisson(lam):
    """P(X >= 1) = 1 - P(0). Pour les buts et les passes."""
    return 1.0 - math.exp(-lam)


def p2plus_negbin(lam, k):
    """P(X >= 2) pour une binomiale négative de moyenne λ et dispersion k.

    POURQUOI pas juste Poisson pour les tirs : la Poisson suppose variance = moyenne.
    Dans la vraie vie, un joueur n'a pas le même contexte chaque soir (temps de PP,
    blowout, changement de trio), donc ses tirs varient PLUS que ça.
    Variance NB = λ + λ²/k. Plus k est grand, plus on se rapproche de Poisson.

    Formules : p = k/(k+λ) ; P(0) = p^k ; P(1) = k·p^k·(1-p)
    """
    p = k / (k + lam)
    p0 = p ** k
    p1 = k * (p ** k) * (1.0 - p)
    return 1.0 - p0 - p1


def pmf_poisson(lam, n_max=15):
    """Liste P(0..n_max). 15 buts couvre 99.9999 % de tout match de hockey."""
    out = [math.exp(-lam)]
    for n in range(1, n_max + 1):
        out.append(out[-1] * lam / n)
    return out


def issue_match(lam_h, lam_a, p_ot_home):
    """Combine deux Poisson indépendantes (buts domicile / visiteur).

    POURQUOI une grille et pas une formule : on veut en un coup P(victoire),
    P(égalité après 60 min) et toute la distribution du total. La grille 16x16
    = 256 cases, c'est instantané.
    L'égalité va en prolongation/fusillade : on la donne au domicile avec
    `p_ot_home` (≈ 52 %, petit avantage historique du dernier changement).
    """
    ph, pa = pmf_poisson(lam_h), pmf_poisson(lam_a)
    win_h = egal = 0.0
    total = [0.0] * 31
    for i, a in enumerate(ph):
        for j, b in enumerate(pa):
            pij = a * b
            if i > j:
                win_h += pij
            elif i == j:
                egal += pij
            total[i + j] += pij
    p_home = win_h + egal * p_ot_home
    over = lambda ligne: 1.0 - sum(total[: int(ligne) + 1])
    return {
        "p_home": p_home,
        "p_egal60": egal,
        "lam_home": lam_h, "lam_away": lam_a,
        "total": lam_h + lam_a,
        "p_over": {"5.5": over(5.5), "6.5": over(6.5)},
    }


# =================================================================================
# v0 : la baseline naïve
# =================================================================================
# POURQUOI garder une version aussi simple : c'est l'étalon. Si v1 ne bat pas v0
# sur des centaines de prédictions, toute la complexité ne sert à rien.

def _moy_par_match(ctx, stat):
    """Moyenne simple de `stat` par match : saison si 3+ matchs, sinon saison passée."""
    cur = ctx["cur"]
    if len(cur) >= 3:
        return sum(g[stat] for g in cur) / len(cur)
    if ctx["prev"]["gp"] > 0:
        return ctx["prev"].get(stat, 0) / ctx["prev"]["gp"]
    return None


def v0_tirs(ctx):
    lam = _moy_par_match(ctx, "shots")
    if lam is None:
        lam = ctx["league"]["pos_rate"][ctx["pos"]] * ctx["league"]["pos_toi"][ctx["pos"]]
    lam = max(lam, 0.05)
    return {"p": p2plus_poisson(lam), "lam": lam}


def v0_joueur(ctx, jeu):
    out = {"tirs": v0_tirs(ctx)}
    for marche, stat in (("buts", "goals"), ("passes", "assists")):
        lam = _moy_par_match(ctx, stat)
        if lam is None:
            lam = ctx["league"]["pos_par_match"][ctx["pos"]][stat]
        lam = max(lam, 0.01)
        out[marche] = {"p": p1plus_poisson(lam), "lam": lam}
    return out


def v0_match(m):
    """Moyenne buts pour/contre de la saison, sans avantage domicile."""
    def moy(t, cle):
        if t["cur_gp"] >= 3:
            return t["cur_" + cle] / t["cur_gp"]
        return t["prev"][cle] / t["prev"]["gp"] if t["prev"]["gp"] else m["mu"]
    h, a = m["home"], m["away"]
    lam_h = max((moy(h, "gf") + moy(a, "ga")) / 2, 0.3)
    lam_a = max((moy(a, "gf") + moy(h, "ga")) / 2, 0.3)
    return issue_match(lam_h, lam_a, 0.5)


# =================================================================================
# v1 : le modèle complet
# =================================================================================

V1 = {
    # --- Tirs (inchangé depuis la publication du 2026-10-17) ---
    # Poids dans le taux "poolé" (stat pondérée / minutes pondérées).
    # Les 10 derniers matchs comptent plus : la forme et le rôle (trio, PP) bougent.
    "w_l10": 1.5,
    "w_reste": 1.0,
    # La saison passée sert d'ancre en début de saison, plafonnée pour ne pas
    # enterrer ce qui se passe cette année.
    "w_prev": 0.35,
    "prev_toi_cap": 900.0,
    # Pseudo-minutes à la moyenne de la position : tire les petits échantillons
    # vers la moyenne (un 4e trio qui a fait 4 tirs en 2 matchs n'est pas un sniper).
    "pseudo_min": 120.0,
    "toi_recent": 0.6,
    "opp_reg_gp": 15.0,
    "home": 1.03,
    "away": 0.97,
    "b2b": 0.96,
    "vs_reg_gp": 12.0,
    "vs_cap": (0.85, 1.15),
    "nb_k": 15.0,

    # --- Buts : λ_buts = λ_tirs × % de tir × facteur gardien adverse ---
    # POURQUOI passer par les tirs : un but = un tir qui rentre. Les tirs sont
    # beaucoup plus stables que les buts (10x plus d'événements), donc on
    # estime les tirs et on multiplie par un % de tir fortement régressé.
    "sh_pseudo_tirs": 80.0,      # 80 tirs fictifs au % de la position
    "sv_reg_gp": 15.0,           # régression du % d'arrêt adverse

    # --- Passes : taux poolé × TOI × (buts attendus de son équipe / moyenne) ---
    # POURQUOI lier aux buts de l'équipe : une passe n'existe que si l'équipe marque.
    # Contre une passoire, tout le monde récolte des points.
    "pa_pseudo_min": 400.0,      # passes = rare et bruité -> shrink plus fort que les tirs

    # --- Matchs ---
    "m_w_l10": 1.5,              # les 10 derniers matchs pèsent plus
    "m_prev_cap": 40.0,          # max 40 matchs de la saison passée
    "m_w_prev": 0.5,
    "m_pseudo_gp": 10.0,         # 10 matchs fictifs à la moyenne de ligue
    "m_home": 1.035,             # domicile : ~+3.5 % de buts, ~-3.5 % pour le visiteur
    "m_away": 0.965,
    "m_b2b_pour": 0.96,          # équipe fatiguée : marque un peu moins...
    "m_b2b_contre": 1.03,        # ...et en accorde un peu plus
    "m_h2h_reg": 8.0,            # duels directs : très bruité, shrink fort
    "m_h2h_cap": (0.88, 1.12),
    "m_p_ot_home": 0.52,
}


# --- Tirs (code identique à la version publiée) ----------------------------------

def taux_poole(ctx, P):
    """Taux de tirs par minute combinant L10, reste de la saison, saison passée
    et une ancre à la moyenne de la position."""
    cur = ctx["cur"]
    l10 = cur[-10:]
    reste = cur[:-10]
    pos_rate = ctx["league"]["pos_rate"][ctx["pos"]]

    prev = ctx["prev"]
    prev_toi = min(prev["toi"], P["prev_toi_cap"])
    prev_rate = prev["shots"] / prev["toi"] if prev["toi"] > 0 else pos_rate

    num = (P["w_l10"] * sum(g["shots"] for g in l10)
           + P["w_reste"] * sum(g["shots"] for g in reste)
           + P["w_prev"] * prev_rate * prev_toi
           + P["pseudo_min"] * pos_rate)
    den = (P["w_l10"] * sum(g["toi"] for g in l10)
           + P["w_reste"] * sum(g["toi"] for g in reste)
           + P["w_prev"] * prev_toi
           + P["pseudo_min"])
    return num / den


def toi_projete(ctx, P):
    cur = ctx["cur"]
    if len(cur) >= 3:
        recent = sum(g["toi"] for g in cur[-5:]) / len(cur[-5:])
        saison = sum(g["toi"] for g in cur) / len(cur)
        return P["toi_recent"] * recent + (1 - P["toi_recent"]) * saison
    if cur:
        moy_cur = sum(g["toi"] for g in cur) / len(cur)
        if ctx["prev"]["gp"] > 0:
            moy_prev = ctx["prev"]["toi"] / ctx["prev"]["gp"]
            return 0.5 * moy_cur + 0.5 * moy_prev
        return moy_cur
    if ctx["prev"]["gp"] > 0:
        return ctx["prev"]["toi"] / ctx["prev"]["gp"]
    return ctx["league"]["pos_toi"][ctx["pos"]]


def facteur_adversaire(ctx, P):
    """Tirs accordés par match de l'adversaire vs la moyenne de la ligue."""
    o = ctx["opp"]
    sa = (o["gp"] * o["sa_cur"] + P["opp_reg_gp"] * o["sa_prev"]) / (o["gp"] + P["opp_reg_gp"])
    return sa / ctx["league"]["sa_avg"]


def facteur_vs(ctx, P):
    """Historique de tirs contre CETTE équipe, avec un shrink fort (3-4 matchs = bruit)."""
    vs = ctx["vs"]
    if vs["toi"] <= 0 or vs["gp"] == 0 or ctx["base_rate"] <= 0:
        return 1.0
    ratio = (vs["shots"] / vs["toi"]) / ctx["base_rate"]
    adj = (vs["gp"] * ratio + P["vs_reg_gp"]) / (vs["gp"] + P["vs_reg_gp"])
    lo, hi = P["vs_cap"]
    return min(max(adj, lo), hi)


def v1_tirs(ctx):
    P = V1
    rate = taux_poole(ctx, P)
    ctx["base_rate"] = rate
    toi = toi_projete(ctx, P)
    f_opp = facteur_adversaire(ctx, P)
    f_lieu = P["home"] if ctx["home"] else P["away"]
    f_b2b = P["b2b"] if ctx["b2b"] else 1.0
    f_vs = facteur_vs(ctx, P)
    lam = max(rate * toi * f_opp * f_lieu * f_b2b * f_vs, 0.05)
    return {
        "p": p2plus_negbin(lam, P["nb_k"]),
        "lam": lam,
        "detail": {
            "rate60": rate * 60, "toi": toi, "f_opp": f_opp,
            "f_lieu": f_lieu, "f_b2b": f_b2b, "f_vs": f_vs,
        },
    }


# --- Buts et passes ---------------------------------------------------------------

def pct_tir(ctx, P):
    """% de tir régressé : 80 tirs fictifs au % moyen de sa position.
    Saison passée comptée à 50 % (le talent de finition est assez stable)."""
    cur, prev = ctx["cur"], ctx["prev"]
    pos_sh = ctx["league"]["pos_sh"][ctx["pos"]]
    buts = sum(g["goals"] for g in cur) + 0.5 * prev.get("goals", 0)
    tirs = sum(g["shots"] for g in cur) + 0.5 * prev["shots"]
    return (buts + P["sh_pseudo_tirs"] * pos_sh) / (tirs + P["sh_pseudo_tirs"])


def facteur_gardien(ctx, P):
    """Buts accordés PAR TIR par l'adversaire vs la ligue.
    > 1 = gardiens qui laissent passer plus que la moyenne."""
    o = ctx["opp"]
    gps = (o["gp"] * o["gaps_cur"] + P["sv_reg_gp"] * o["gaps_prev"]) / (o["gp"] + P["sv_reg_gp"])
    return gps / ctx["league"]["gaps_avg"]


def taux_passes(ctx, P):
    """Même logique de pooling que les tirs, appliquée aux passes."""
    cur, prev = ctx["cur"], ctx["prev"]
    l10, reste = cur[-10:], cur[:-10]
    pos_rate = ctx["league"]["pos_pa_rate"][ctx["pos"]]
    prev_toi = min(prev["toi"], P["prev_toi_cap"])
    prev_rate = prev.get("assists", 0) / prev["toi"] if prev["toi"] > 0 else pos_rate
    num = (P["w_l10"] * sum(g["assists"] for g in l10)
           + P["w_reste"] * sum(g["assists"] for g in reste)
           + P["w_prev"] * prev_rate * prev_toi
           + P["pa_pseudo_min"] * pos_rate)
    den = (P["w_l10"] * sum(g["toi"] for g in l10)
           + P["w_reste"] * sum(g["toi"] for g in reste)
           + P["w_prev"] * prev_toi
           + P["pa_pseudo_min"])
    return num / den


def v1_joueur(ctx, jeu):
    P = V1
    tirs = v1_tirs(ctx)

    sh = pct_tir(ctx, P)
    f_g = facteur_gardien(ctx, P)
    lam_b = max(tirs["lam"] * sh * f_g, 0.01)

    toi = tirs["detail"]["toi"]
    f_equipe = jeu["lam_pour"] / ctx["league"]["mu"] if jeu else 1.0
    lam_p = max(taux_passes(ctx, P) * toi * f_equipe, 0.01)

    return {
        "tirs": tirs,
        "buts": {"p": p1plus_poisson(lam_b), "lam": lam_b,
                 "detail": {"sh": sh, "f_gardien": f_g}},
        "passes": {"p": p1plus_poisson(lam_p), "lam": lam_p,
                   "detail": {"f_equipe": f_equipe}},
    }


# --- Matchs ----------------------------------------------------------------------

def force(t, cle, mu, P):
    """Buts pour (cle='gf') ou contre ('ga') par match, poolés comme les tirs :
    10 derniers (x1.5) + reste de la saison + saison passée plafonnée + ancre ligue."""
    jeux = t["cur"]
    l10, reste = jeux[-10:], jeux[:-10]
    prev = t["prev"]
    n_prev = min(prev["gp"], P["m_prev_cap"])
    prev_pg = prev[cle] / prev["gp"] if prev["gp"] else mu
    num = (P["m_w_l10"] * sum(g[cle] for g in l10) + sum(g[cle] for g in reste)
           + P["m_w_prev"] * n_prev * prev_pg + P["m_pseudo_gp"] * mu)
    den = (P["m_w_l10"] * len(l10) + len(reste)
           + P["m_w_prev"] * n_prev + P["m_pseudo_gp"])
    return num / den


def facteur_h2h(t, opp, att, P):
    """Ce que tu demandais : les derniers duels entre les deux équipes.
    On compare les buts marqués contre CET adversaire à l'attaque normale,
    puis on shrink fort : 2-4 matchs par an, c'est surtout du bruit."""
    gp, gf = t["h2h"]["gp"], t["h2h"]["gf"]
    if gp == 0 or att <= 0:
        return 1.0
    ratio = (gf / gp) / att
    adj = (gp * ratio + P["m_h2h_reg"]) / (gp + P["m_h2h_reg"])
    lo, hi = P["m_h2h_cap"]
    return min(max(adj, lo), hi)


def v1_match(m):
    P = V1
    mu = m["mu"]
    h, a = m["home"], m["away"]
    att_h, def_h = force(h, "gf", mu, P), force(h, "ga", mu, P)
    att_a, def_a = force(a, "gf", mu, P), force(a, "ga", mu, P)

    # Modèle multiplicatif classique : attaque × défense adverse / moyenne
    lam_h = att_h * def_a / mu * P["m_home"]
    lam_a = att_a * def_h / mu * P["m_away"]

    if h["b2b"]:
        lam_h *= P["m_b2b_pour"]; lam_a *= P["m_b2b_contre"]
    if a["b2b"]:
        lam_a *= P["m_b2b_pour"]; lam_h *= P["m_b2b_contre"]

    f_h2h_h = facteur_h2h(h, a["abbrev"], att_h, P)
    f_h2h_a = facteur_h2h(a, h["abbrev"], att_a, P)
    lam_h *= f_h2h_h
    lam_a *= f_h2h_a

    out = issue_match(max(lam_h, 0.3), max(lam_a, 0.3), P["m_p_ot_home"])
    out["detail"] = {"att_h": att_h, "def_h": def_h, "att_a": att_a, "def_a": def_a,
                     "h2h_h": f_h2h_h, "h2h_a": f_h2h_a, "mu": mu}
    return out


# =================================================================================
# Registre
# =================================================================================
# Ordre = ordre d'affichage. La DERNIÈRE est la version "principale" sur la page.
VERSIONS = {
    "v0": {"joueur": v0_joueur, "match": v0_match,
           "desc": "Baseline : moyennes de la saison, aucun ajustement"},
    "v1": {"joueur": v1_joueur, "match": v1_match,
           "desc": "Forme L10, temps de glace, adversaire, gardiens, domicile, back-to-back, duels directs"},
}
VERSION_PRINCIPALE = list(VERSIONS)[-1]
MARCHES = ["tirs", "buts", "passes"]
