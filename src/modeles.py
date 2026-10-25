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


# Seuils de tirs offerts : marché -> nombre de tirs minimum.
# "tirs" (2+) garde son nom d'origine pour ne pas casser l'historique.
SEUILS_TIRS = {"tirs1": 1, "tirs": 2, "tirs3": 3, "tirs4": 4}


def p_seuil(lam, seuil, k=None):
    """P(X >= seuil). k=None : Poisson ; sinon binomiale négative (dispersion k).

    On additionne P(0)..P(seuil-1) par récurrence (pas de factorielles géantes) :
      Poisson : P(n) = P(n-1) × λ / n
      Bin. nég. : P(0) = p^k avec p = k/(k+λ) ; P(n) = P(n-1) × (k+n-1)/n × (1-p)
    """
    if k is None:
        pn = math.exp(-lam)
        cumul = pn
        for n in range(1, seuil):
            pn *= lam / n
            cumul += pn
    else:
        p = k / (k + lam)
        pn = p ** k
        cumul = pn
        for n in range(1, seuil):
            pn *= (k + n - 1) / n * (1 - p)
            cumul += pn
    return 1.0 - cumul


def ajouter_seuils(pv, k):
    """Ajoute tirs1/tirs3/tirs4 à la prédiction d'une version, à partir de SES
    tirs attendus (λ) et de SA distribution.

    POURQUOI c'est honnête même sur de vieilles prédictions : λ a été calculé et
    sauvé AVANT le match. 1+, 3+ et 4+ sont juste d'autres lectures de la même
    courbe, comme lire 25 % ou 75 % sur la même jauge. Rien d'après-match n'entre.
    Retourne True si quelque chose a été ajouté."""
    if "tirs" in pv:
        lam = pv["tirs"]["lam"]
    elif "lam" in pv and "p" in pv:     # ancien format du 17 octobre : tirs seulement
        lam = pv["lam"]
    else:
        return False
    ajoute = False
    for m, seuil in SEUILS_TIRS.items():
        if m != "tirs" and m not in pv:
            pv[m] = {"p": round(p_seuil(lam, seuil, k), 4), "lam": round(lam, 4)}
            ajoute = True
    return ajouter_marches(pv) or ajoute


def _au_moins_poisson(lam, s):
    """P(X >= s) pour une loi de Poisson de moyenne lam."""
    p, cumul = math.exp(-lam), 0.0
    for n in range(s):
        cumul += p
        p *= lam / (n + 1)
    return max(0.0, 1 - cumul)


# Marchés dérivés (24 oct.) : 2+ et 3+ buts, 2+ passes, 1+ et 2+ points.
# Même principe que les seuils de tirs : d'autres lectures des buts et passes attendus
# (λ sauvés AVANT le match), donc honnête même sur les vieilles prédictions.
# Points = buts + passes : λ_points = λ_buts + λ_passes (Poisson). Limite : buts et
# passes d'un même joueur montent ensemble les bons soirs, donc 2+ points est un peu sous-estimé.
def ajouter_marches(pv):
    ajoute = False
    lb = (pv.get("buts") or {}).get("lam")
    la = (pv.get("passes") or {}).get("lam")
    nouveaux = {}
    if lb is not None:
        nouveaux["buts2"] = (_au_moins_poisson(lb, 2), lb)
        nouveaux["buts3"] = (_au_moins_poisson(lb, 3), lb)
    if la is not None:
        nouveaux["passes2"] = (_au_moins_poisson(la, 2), la)
    if lb is not None and la is not None:
        nouveaux["pts1"] = (_au_moins_poisson(lb + la, 1), lb + la)
        nouveaux["pts2"] = (_au_moins_poisson(lb + la, 2), lb + la)
    for m, (p, lam) in nouveaux.items():
        if m not in pv:
            pv[m] = {"p": round(p, 4), "lam": round(lam, 4)}
            ajoute = True
    return ajoute


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
# Modèle paramétrique (auto-upgrade)
# =================================================================================
# POURQUOI : pour que l'algo puisse s'améliorer tout seul, il doit pouvoir REJOUER
# n'importe quel soir passé avec d'autres paramètres, sans rappeler l'API. On
# enregistre donc avec chaque prédiction ses "features" : les sommes brutes que
# v1 utilise (tirs, minutes, adversaire...). modele_joueur_param(f, V1) redonne
# exactement v1 (vérifié par test). Avec d'autres paramètres, ça donne un candidat.

def features_joueur(ctx, jeu):
    """Contexte brut d'un joueur avant le match. Clés courtes : stocké des milliers de fois."""
    cur = ctx["cur"]
    l10, reste = cur[-10:], cur[:-10]
    s = lambda gs, k: sum(g[k] for g in gs)
    bloc = lambda gs: [s(gs, "shots"), s(gs, "goals"), s(gs, "assists"), s(gs, "toi"), len(gs)]
    pv, o, lg, vs = ctx["prev"], ctx["opp"], ctx["league"], ctx["vs"]
    pos = ctx["pos"]
    return {
        "pos": pos, "home": ctx["home"], "b2b": ctx["b2b"],
        "l10": bloc(l10), "rs": bloc(reste),
        "pv": [pv["shots"], pv.get("goals", 0), pv.get("assists", 0), pv["toi"], pv["gp"]],
        "t5": s(cur[-5:], "toi") / len(cur[-5:]) if cur else None,
        "tsa": s(cur, "toi") / len(cur) if cur else None,
        "n": len(cur),
        "op": [o["gp"], o["sa_cur"], o["sa_prev"], o["gaps_cur"], o["gaps_prev"]],
        "lg": [lg["sa_avg"], lg["gaps_avg"], lg["pos_rate"][pos], lg["pos_sh"][pos],
               lg["pos_pa_rate"][pos], lg["pos_toi"][pos], lg["mu"]],
        "vs": [vs["shots"], vs["toi"], vs["gp"]],
        "eq": jeu["lam_pour"] if jeu else None,
    }


def _taux(f, P, i, pseudo):
    """Taux par minute poolé (i = 0 tirs, 2 passes), même formule que v1."""
    l10, rs, pv = f["l10"], f["rs"], f["pv"]
    pos_rate = f["lg"][2] if i == 0 else f["lg"][4]
    prev_toi = min(pv[3], P["prev_toi_cap"])
    prev_rate = pv[i] / pv[3] if pv[3] > 0 else pos_rate
    num = P["w_l10"] * l10[i] + P["w_reste"] * rs[i] + P["w_prev"] * prev_rate * prev_toi + pseudo * pos_rate
    den = P["w_l10"] * l10[3] + P["w_reste"] * rs[3] + P["w_prev"] * prev_toi + pseudo
    return num / den


def _toi(f, P):
    pv = f["pv"]
    if f["n"] >= 3:
        return P["toi_recent"] * f["t5"] + (1 - P["toi_recent"]) * f["tsa"]
    if f["n"]:
        return 0.5 * f["tsa"] + 0.5 * (pv[3] / pv[4]) if pv[4] > 0 else f["tsa"]
    if pv[4] > 0:
        return pv[3] / pv[4]
    return f["lg"][5]


def modele_joueur_param(f, P):
    """Les 3 marchés d'un joueur à partir de ses features et d'un jeu de paramètres.
    Les clés scale_* (absentes de V1 -> 1.0) servent au tuner pour corriger un biais
    global trouvé par le détecteur de patterns (ex. défenseurs surestimés)."""
    g = lambda k, d=1.0: P.get(k, d)
    rate = _taux(f, P, 0, P["pseudo_min"])
    toi = _toi(f, P)
    o, lg = f["op"], f["lg"]
    f_opp = (o[0] * o[1] + P["opp_reg_gp"] * o[2]) / (o[0] + P["opp_reg_gp"]) / lg[0]
    f_lieu = P["home"] if f["home"] else P["away"]
    f_b2b = P["b2b"] if f["b2b"] else 1.0
    vs = f["vs"]
    if vs[1] <= 0 or vs[2] == 0 or rate <= 0:
        f_vs = 1.0
    else:
        ratio = (vs[0] / vs[1]) / rate
        adj = (vs[2] * ratio + P["vs_reg_gp"]) / (vs[2] + P["vs_reg_gp"])
        f_vs = min(max(adj, P["vs_cap"][0]), P["vs_cap"][1])
    echelle_t = g("scale_tirs") * (g("scale_tirs_D") if f["pos"] == "D" else 1.0)
    lam_t = max(rate * toi * f_opp * f_lieu * f_b2b * f_vs * echelle_t, 0.05)

    l10, rs, pv = f["l10"], f["rs"], f["pv"]
    sh = (l10[1] + rs[1] + 0.5 * pv[1] + P["sh_pseudo_tirs"] * lg[3]) / (l10[0] + rs[0] + 0.5 * pv[0] + P["sh_pseudo_tirs"])
    f_g = (o[0] * o[3] + P["sv_reg_gp"] * o[4]) / (o[0] + P["sv_reg_gp"]) / lg[1]
    lam_b = max(lam_t * sh * f_g * g("scale_buts"), 0.01)

    f_eq = f["eq"] / lg[6] if f["eq"] else 1.0
    lam_p = max(_taux(f, P, 2, P["pa_pseudo_min"]) * toi * f_eq * g("scale_passes"), 0.01)
    out_tirs = {m: {"p": p_seuil(lam_t, seuil, P["nb_k"]), "lam": lam_t} for m, seuil in SEUILS_TIRS.items()}
    out_tirs["tirs"]["p"] = p2plus_negbin(lam_t, P["nb_k"])   # même formule exacte que v1
    return {
        **out_tirs,
        "buts": {"p": p1plus_poisson(lam_b), "lam": lam_b},
        "passes": {"p": p1plus_poisson(lam_p), "lam": lam_p},
    }


def features_match(m):
    """Contexte brut d'un match (les deux équipes) avant la mise au jeu."""
    def cote(t):
        jeux = t["cur"]
        l10, reste = jeux[-10:], jeux[:-10]
        s = lambda gs, k: sum(x[k] for x in gs)
        pv = t["prev"]
        return {"l10": [s(l10, "gf"), s(l10, "ga"), len(l10)], "rs": [s(reste, "gf"), s(reste, "ga"), len(reste)],
                "pv": [pv["gp"], pv["gf"], pv["ga"]], "h2h": [t["h2h"]["gf"], t["h2h"]["gp"]], "b2b": t["b2b"]}
    return {"mu": m["mu"], "h": cote(m["home"]), "a": cote(m["away"])}


def modele_match_param(f, P):
    """Même modèle que v1_match, rejouable à partir des features."""
    mu = f["mu"]
    def force(c, i):   # i = 0 buts pour, 1 buts contre
        n_prev = min(c["pv"][0], P["m_prev_cap"])
        prev_pg = c["pv"][1 + i] / c["pv"][0] if c["pv"][0] else mu
        num = P["m_w_l10"] * c["l10"][i] + c["rs"][i] + P["m_w_prev"] * n_prev * prev_pg + P["m_pseudo_gp"] * mu
        den = P["m_w_l10"] * c["l10"][2] + c["rs"][2] + P["m_w_prev"] * n_prev + P["m_pseudo_gp"]
        return num / den
    def h2h(c, att):
        gf, gp = c["h2h"]
        if gp == 0 or att <= 0:
            return 1.0
        adj = (gp * ((gf / gp) / att) + P["m_h2h_reg"]) / (gp + P["m_h2h_reg"])
        return min(max(adj, P["m_h2h_cap"][0]), P["m_h2h_cap"][1])
    h, a = f["h"], f["a"]
    att_h, def_h, att_a, def_a = force(h, 0), force(h, 1), force(a, 0), force(a, 1)
    lam_h = att_h * def_a / mu * P["m_home"]
    lam_a = att_a * def_h / mu * P["m_away"]
    if h["b2b"]:
        lam_h *= P["m_b2b_pour"]; lam_a *= P["m_b2b_contre"]
    if a["b2b"]:
        lam_a *= P["m_b2b_pour"]; lam_h *= P["m_b2b_contre"]
    lam_h *= h2h(h, att_h) * P.get("scale_match", 1.0)
    lam_a *= h2h(a, att_a) * P.get("scale_match", 1.0)
    return issue_match(max(lam_h, 0.3), max(lam_a, 0.3), P["m_p_ot_home"])


# =================================================================================
# v2 : tentatives de tirs, avantage numérique, indiscipline, état de match,
#      calibration, mélange avec le marché
# =================================================================================
# Même ossature que v1 (rejouable à partir des features, donc tunable), avec en plus
# les features "x" (voir donnees_v2.py et run.py) :
#   x["a"]  = [att_l10, toi_att_l10, sog_att_l10, att_rs, toi_att_rs, sog_att_rs, att_prev, gp_prev]
#   x["pp"] = [pp_l10, n_pp_l10, pp_rs, n_pp_rs, pp_prev, gp_prev, pp_l5]   (minutes)
#   x["osh"] = minutes de PP par match accordées par l'adversaire (régressées)
#   x["lpp"] = moyenne de ligue des minutes de PP par équipe par match
#   x["pw"]  = % de victoire de l'équipe du joueur (modèle de match v1)
#   x["lr"]  = ratio tirs cadrés / tentatives moyen de sa position
# f["mkt"] (optionnel) = probabilités implicites du marché (cotes US) par marché.

V2 = {
    **V1,
    # Minutes « effectives » : une minute en PP vaut (1 + k_pp) minutes à 5 c. 5 pour les tirs
    "k_pp": 1.0,
    # Poids des tentatives dans le taux (0 = tirs cadrés seulement, 1 = tentatives seulement)
    "w_att": 0.5,
    "ratio_pseudo": 60.0,      # tentatives fictives au ratio moyen de la position
    # Indiscipline : (min de PP accordées par l'adversaire / moyenne) ^ pen_exp
    "pen_exp": 1.0,
    # État de match : une équipe qui tire de l'arrière tire plus (score effects)
    "score_s": 0.12,
    # Passes : même logique de minutes effectives
    "k_pp_passes": 1.5,
    # Mélange avec le marché (en logit) : p = σ(w·logit(p_modèle) + (1-w)·logit(p_marché))
    "w_mod_joueur": 0.55,
    "w_mod_match": 0.45,
    "vig_props": 1.08,         # marge approx. d'une prop (on ne voit que le côté « plus »)
    # Calibration (Platt) de la victoire : p' = σ(a·logit(p) + b)
    "cal_a": 1.0,
    "cal_b": 0.0,
    # Match par les tirs (xG simplifié) plutôt que par les buts seulement
    "mt_pseudo_gp": 8.0,
    "mt_sh_pseudo": 500.0,     # tirs fictifs au % de tir moyen de la ligue
    "mt_w_buts": 0.60,         # part du vieux modèle par buts dans le λ final (prudent au départ)
    # --- Ajouts du 23 octobre. Valeurs NEUTRES : v2.x et v3.x donnent exactement les
    # mêmes % qu'avant. C'est l'évolution (mutations, tuner) qui décide s'ils servent. ---
    # Biais de patinoire : certains marqueurs officiels comptent plus de tirs que d'autres.
    # λ × (facteur de l'aréna)^w_rink. 0 = ignoré, 1 = facteur complet.
    "w_rink": 0.0,
    # Prior selon le rôle : le taux « de départ » d'un joueur à 22 min n'est pas celui d'un
    # joueur à 10 min. taux_prior × (TOI projeté / TOI moyen)^prior_toi_exp. 0 = même prior pour tous.
    "prior_toi_exp": 0.0,
    # Recalibration des % du modèle (avant le marché) : p' = σ(a·logit(p) + b).
    # b < 0 corrige un modèle qui voit trop haut ; a < 1 calme les extrêmes.
    "cal_t_a": 1.0, "cal_t_b": 0.0,      # tirs 1+ à 4+ (même correction pour les 4 : l'ordre est gardé)
    "cal_bu_a": 1.0, "cal_bu_b": 0.0,    # but
    "cal_pa_a": 1.0, "cal_pa_b": 0.0,    # passe
    # Dispersion propre à chaque joueur (24 oct.) : certains gars varient beaucoup plus que
    # d'autres d'un match à l'autre (souvent 0-1 tir, ou 4+). On mélange la dispersion de la
    # ligue avec celle de SES 10 derniers matchs. 0 = même dispersion pour tous.
    "w_disp": 0.0,
    # Robustesse de l'adversaire (25 oct., idée de Mitch) : certaines équipes frappent beaucoup.
    # w_phys : effet moyen sur tout le monde, λ × indice^w_phys (négatif = on tire moins contre eux).
    # w_phys_j : effet propre au joueur, selon SON rythme passé contre les équipes robustes.
    "w_phys": 0.0, "w_phys_j": 0.0,
}


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + math.exp(-z))


def melanger(p_mod, p_mkt, w):
    """Combinaison en logit : on fait une moyenne pondérée des « cotes » plutôt que
    des %, parce qu'un 90 % et un 95 % sont très différents (10 % vs 5 % de risque)."""
    if p_mkt is None:
        return p_mod
    return _sig(w * _logit(p_mod) + (1 - w) * _logit(p_mkt))


def _dispersion_joueur(serie, k_ligue, w):
    """Dispersion (k de la binomiale négative) d'un joueur. Petit k = matchs très inégaux.
    k du joueur estimé sur ses derniers matchs (variance vs moyenne), puis mélangé avec la
    ligue, d'autant plus que la série est longue. On mélange 1/k (l'excès de variance)."""
    if not w or not serie or len(serie) < 5:
        return k_ligue
    n = len(serie)
    m = sum(serie) / n
    v = sum((x - m) ** 2 for x in serie) / (n - 1)
    k_j = m * m / (v - m) if v > m * 1.02 and m > 0 else 500.0
    poids = w * n / (n + 10)
    inv = (1 - poids) / k_ligue + poids / max(k_j, 1.0)
    return min(max(1 / inv, 1.0), 500.0)


def modele_joueur_v2(f, P):
    x = f.get("x")
    if not x:
        # Vieille prédiction sans données v2 : même chose que v1 (mais calibrée/mélangée)
        out = modele_joueur_param(f, P)
    else:
        g = lambda k, d=1.0: P.get(k, d)
        l10, rs, pv, lg = f["l10"], f["rs"], f["pv"], f["lg"]
        a, pp = x["a"], x["pp"]
        k = P["k_pp"]
        toi = _toi(f, P)
        # Prior selon le rôle (neutre si prior_toi_exp = 0)
        e_prior = g("prior_toi_exp", 0.0)
        f_role = (toi / lg[5]) ** e_prior if e_prior and toi > 0 and lg[5] else 1.0
        pos_rate = lg[2] * f_role

        # --- Temps de PP projeté, ajusté à l'indiscipline de l'adversaire ---
        pp_sa = (pp[0] + pp[2]) / (pp[1] + pp[3]) if (pp[1] + pp[3]) else None
        pp_prev = pp[4] / pp[5] if pp[5] else None
        base_pp = [v for v in (pp_sa, pp_prev) if v is not None]
        pp_moy = 0.6 * (pp[6] if pp[6] is not None else base_pp[0]) + 0.4 * base_pp[0] if base_pp else 0.0
        f_pen = (x["osh"] / x["lpp"]) ** P["pen_exp"] if x.get("lpp") else 1.0
        pp_proj = pp_moy * f_pen

        # --- Taux de tirs par minute effective (cadrés + tentatives) ---
        em = lambda toi_, ppm, coef: toi_ + coef * ppm
        prev_toi = min(pv[3], P["prev_toi_cap"])
        part_prev = prev_toi / pv[3] if pv[3] else 0
        em_l10, em_rs = em(l10[3], pp[0], k), em(rs[3], pp[2], k)
        em_pv = em(pv[3], pp[4], k) * part_prev
        prev_rate = pv[0] / em(pv[3], pp[4], k) if pv[3] else pos_rate
        num = P["w_l10"] * l10[0] + P["w_reste"] * rs[0] + P["w_prev"] * prev_rate * em_pv + P["pseudo_min"] * pos_rate
        den = P["w_l10"] * em_l10 + P["w_reste"] * em_rs + P["w_prev"] * em_pv + P["pseudo_min"]
        rate_sog = num / den

        ratio_ligue = x.get("lr") or 0.53
        sog_a, att_a = a[2] + a[5] + 0.5 * pv[0], a[0] + a[3] + 0.5 * (a[6] or 0)
        ratio = (sog_a + P["ratio_pseudo"] * ratio_ligue) / (att_a + P["ratio_pseudo"])
        # Les minutes effectives des matchs où on connaît les tentatives
        f_l10 = a[1] / l10[3] if l10[3] else 0
        f_rs = a[4] / rs[3] if rs[3] else 0
        em_att_l10, em_att_rs = em(a[1], pp[0] * f_l10, k), em(a[4], pp[2] * f_rs, k)
        prev_att_rate = (a[6] / em(pv[3], pp[4], k)) if (a[6] and pv[3]) else pos_rate / ratio_ligue
        num_a = P["w_l10"] * a[0] + P["w_reste"] * a[3] + P["w_prev"] * prev_att_rate * em_pv + P["pseudo_min"] * pos_rate / ratio_ligue
        den_a = P["w_l10"] * em_att_l10 + P["w_reste"] * em_att_rs + P["w_prev"] * em_pv + P["pseudo_min"]
        rate_att = num_a / den_a
        rate = (1 - P["w_att"]) * rate_sog + P["w_att"] * rate_att * ratio

        em_proj = em(toi, pp_proj, k)
        o = f["op"]
        f_opp = (o[0] * o[1] + P["opp_reg_gp"] * o[2]) / (o[0] + P["opp_reg_gp"]) / lg[0]
        f_lieu = P["home"] if f["home"] else P["away"]
        f_b2b = P["b2b"] if f["b2b"] else 1.0
        vs = f["vs"]
        rate_toi = _taux(f, P, 0, P["pseudo_min"])
        if vs[1] <= 0 or vs[2] == 0 or rate_toi <= 0:
            f_vs = 1.0
        else:
            adj = (vs[2] * ((vs[0] / vs[1]) / rate_toi) + P["vs_reg_gp"]) / (vs[2] + P["vs_reg_gp"])
            f_vs = min(max(adj, P["vs_cap"][0]), P["vs_cap"][1])
        f_score = 1 + P["score_s"] * (0.5 - x["pw"]) if x.get("pw") is not None else 1.0
        echelle = g("scale_tirs") * (g("scale_tirs_D") if f["pos"] == "D" else 1.0)
        # Patinoire : change les tirs COMPTÉS, pas les buts. Les buts partent donc du λ sans ce facteur.
        f_rink = x.get("rk", 1.0) ** g("w_rink", 0.0) if x.get("rk") else 1.0
        ph = x.get("ph")
        f_phys = 1.0
        if ph and (g("w_phys", 0.0) or g("w_phys_j", 0.0)):
            oi, ratio_r, n_r = ph
            f_phys = oi ** g("w_phys", 0.0)
            if oi > 1.0:   # ce soir contre une équipe robuste : son historique contre ce genre d'équipe
                f_phys *= 1 + g("w_phys_j", 0.0) * n_r / (n_r + 8) * (ratio_r - 1) * min(1.0, (oi - 1) / 0.10)
            f_phys = min(1.3, max(0.7, f_phys))
        lam_t0 = max(rate * em_proj * f_opp * f_lieu * f_b2b * f_vs * f_score * echelle * f_phys, 0.05)
        lam_t = lam_t0 * f_rink

        sh = (l10[1] + rs[1] + 0.5 * pv[1] + P["sh_pseudo_tirs"] * lg[3]) / (l10[0] + rs[0] + 0.5 * pv[0] + P["sh_pseudo_tirs"])
        f_g = (o[0] * o[3] + P["sv_reg_gp"] * o[4]) / (o[0] + P["sv_reg_gp"]) / lg[1]
        lam_b = max(lam_t0 * sh * f_g * g("scale_buts"), 0.01)

        # Passes par minute effective (le PP génère beaucoup de passes)
        kp = P["k_pp_passes"]
        em_p = lambda t_, ppm: t_ + kp * ppm
        prev_rate_p = pv[2] / em_p(pv[3], pp[4]) if pv[3] else lg[4]
        em_pv_p = em_p(pv[3], pp[4]) * part_prev
        num_p = P["w_l10"] * l10[2] + P["w_reste"] * rs[2] + P["w_prev"] * prev_rate_p * em_pv_p + P["pa_pseudo_min"] * lg[4]
        den_p = P["w_l10"] * em_p(l10[3], pp[0]) + P["w_reste"] * em_p(rs[3], pp[2]) + P["w_prev"] * em_pv_p + P["pa_pseudo_min"]
        f_eq = f["eq"] / lg[6] if f["eq"] else 1.0
        lam_p = max(num_p / den_p * em_p(toi, pp_proj) * f_eq * g("scale_passes"), 0.01)

        k_eff = _dispersion_joueur(f.get("s10"), P["nb_k"], g("w_disp", 0.0))
        out = {m: {"p": p_seuil(lam_t, seuil, k_eff), "lam": lam_t} for m, seuil in SEUILS_TIRS.items()}
        out["buts"] = {"p": p1plus_poisson(lam_b), "lam": lam_b}
        out["passes"] = {"p": p1plus_poisson(lam_p), "lam": lam_p}
        # Détail pour « Pourquoi ce % » sur la page. N'affecte AUCUNE probabilité :
        # c'est juste les morceaux du calcul, pour voir d'où vient le chiffre.
        # r_sog / r_att : les deux estimés du taux (par les cadrés seulement, et par
        # les tentatives × % qui se rendent au filet), mélangés selon w_att.
        out["tirs"]["detail"] = {"rate60": rate * 60, "toi": toi, "f_opp": f_opp, "f_lieu": f_lieu, "f_b2b": f_b2b,
                                 "f_vs": f_vs, "pp": pp_proj, "f_pen": f_pen, "f_score": f_score,
                                 "r_sog": rate_sog * 60, "r_att": rate_att * ratio * 60, "ratio": ratio,
                                 "w_att": P["w_att"], "k_pp": k}
        if f_rink != 1.0:
            out["tirs"]["detail"]["f_rink"] = f_rink
        if f_role != 1.0:
            out["tirs"]["detail"]["f_role"] = f_role
        if f_phys != 1.0:
            out["tirs"]["detail"]["f_phys"] = f_phys
        out["buts"]["detail"] = {"sh": sh, "f_gardien": f_g, "pp": pp_proj}
        out["passes"]["detail"] = {"f_equipe": f_eq, "pp": pp_proj, "k_pp": kp}

    # Recalibration du modèle, avant le marché (identité avec les valeurs par défaut)
    for marches, ca, cb in ((SEUILS_TIRS, "cal_t_a", "cal_t_b"), (("buts",), "cal_bu_a", "cal_bu_b"),
                            (("passes",), "cal_pa_a", "cal_pa_b")):
        A, B = P.get(ca, 1.0), P.get(cb, 0.0)
        if A != 1.0 or B != 0.0:
            for m in marches:
                if m in out:
                    out[m]["p"] = _sig(A * _logit(out[m]["p"]) + B)

    # --- Mélange avec le marché, marché par marché ---
    mkt = f.get("mkt") or {}
    for m, implicite in mkt.items():
        if m in out and implicite:
            p_mkt = min(implicite / P["vig_props"], 0.995)
            out[m]["p0"] = out[m]["p"]
            out[m]["p"] = melanger(out[m]["p"], p_mkt, P["w_mod_joueur"])
            # Pour la page : % du marché (sans la marge du bookie) et poids du modèle
            out[m]["mkt"], out[m]["w"] = p_mkt, P["w_mod_joueur"]
            if m == "tirs" and "detail" in out[m]:
                out[m]["detail"]["mkt"] = p_mkt
    ajouter_marches(out)
    return out


def modele_match_v2(f, P):
    """Buts attendus par les TIRS (plus stables que les buts), mélangés avec v1,
    puis calibration de la victoire et mélange avec le marché."""
    base = modele_match_param(f, P)
    x = f.get("x")
    if x:
        lg_sf, lg_sh = x["lg"][0], x["lg"][1]
        def pool(c, i, moy):   # i : 0 tirs pour, 1 tirs contre, 2 buts pour, 3 buts contre
            l, r, pv = c["l10"], c["rs"], c["pv"]
            n_prev = min(pv[0], P["m_prev_cap"]) if pv else 0
            prev_pg = pv[1 + i] / pv[0] if pv and pv[0] else moy
            num = P["m_w_l10"] * l[i] + r[i] + P["m_w_prev"] * n_prev * prev_pg + P["mt_pseudo_gp"] * moy
            den = P["m_w_l10"] * l[4] + r[4] + P["m_w_prev"] * n_prev + P["mt_pseudo_gp"]
            return num / den
        def eq(c):
            sf, sa = pool(c, 0, lg_sf), pool(c, 1, lg_sf)
            gf, ga = pool(c, 2, lg_sf * lg_sh), pool(c, 3, lg_sf * lg_sh)
            # % de tir et buts accordés par tir, régressés vers la ligue avec
            # mt_sh_pseudo tirs fictifs (ramenés « par match » comme les moyennes poolées)
            ps = P["mt_sh_pseudo"] / 82
            return sf, sa, (gf + ps * lg_sh) / (sf + ps), (ga + ps * lg_sh) / (sa + ps)
        sf_h, sa_h, sh_h, gaps_h = eq(x["h"])
        sf_a, sa_a, sh_a, gaps_a = eq(x["a"])
        tirs_h = sf_h * sa_a / lg_sf
        tirs_a = sf_a * sa_h / lg_sf
        lam_h = tirs_h * sh_h * gaps_a / lg_sh * P["m_home"]
        lam_a = tirs_a * sh_a * gaps_h / lg_sh * P["m_away"]
        if f["h"]["b2b"]:
            lam_h *= P["m_b2b_pour"]; lam_a *= P["m_b2b_contre"]
        if f["a"]["b2b"]:
            lam_a *= P["m_b2b_pour"]; lam_h *= P["m_b2b_contre"]
        w = P["mt_w_buts"]
        lam_h = (1 - w) * lam_h * P.get("scale_match", 1.0) + w * base["lam_home"]
        lam_a = (1 - w) * lam_a * P.get("scale_match", 1.0) + w * base["lam_away"]
        base = issue_match(max(lam_h, 0.3), max(lam_a, 0.3), P["m_p_ot_home"])
    # Calibration de la victoire (Platt), puis marché
    base["p_home"] = _sig(P["cal_a"] * _logit(base["p_home"]) + P["cal_b"])
    mkt = f.get("mkt") or {}
    if mkt.get("home"):
        base["p_home0"] = base["p_home"]
        base["p_home"] = melanger(base["p_home"], mkt["home"], P["w_mod_match"])
    for ligne in ("5.5", "6.5"):
        if mkt.get("o" + ligne):
            base["p_over"][ligne] = melanger(base["p_over"][ligne], mkt["o" + ligne], P["w_mod_match"])
    return base


# Familles de modèles rejouables (le tuner et les versions auto s'en servent)
FAMILLES = {
    "v1": {"joueur": modele_joueur_param, "match": modele_match_param, "P": V1},
    "v2": {"joueur": modele_joueur_v2, "match": modele_match_v2, "P": V2},
}


# =================================================================================
# Registre
# =================================================================================
# Ordre = ordre d'affichage. La DERNIÈRE est la version "principale" sur la page.
VERSIONS = {
    "v0": {"joueur": v0_joueur, "match": v0_match, "nb_k": None,
           "desc": "Baseline : moyennes de la saison, aucun ajustement"},
    "v1": {"joueur": v1_joueur, "match": v1_match, "nb_k": V1["nb_k"],
           "desc": "Forme L10, temps de glace, adversaire, gardiens, domicile, back-to-back, duels directs"},
    # Génération 2 : 5 « philosophies » fixées d'avance (pas entraînées sur l'historique),
    # donc rejouées honnêtement sur toute la saison, comme v1. Voir GEN2 plus bas.
}
# Génération 2 : même modèle (famille v2), 5 réglages qui testent chacun une idée.
# v2.1 est la référence ; les autres poussent UNE direction à la fois, pour qu'on
# apprenne quelque chose de lisible du gagnant (« le marché compte plus », etc.).
GEN2 = {
    "v2.1": ({}, "Référence v2 : tentatives + PP + indiscipline + état de match + marché 45-55 %"),
    "v2.2": ({"w_mod_joueur": 0.35, "w_mod_match": 0.30},
             "Fait plus confiance au marché (65-70 % du poids aux cotes US)"),
    "v2.3": ({"w_mod_joueur": 0.80, "w_mod_match": 0.70, "w_att": 0.75},
             "Fait plus confiance au modèle et aux tentatives de tirs, moins au marché"),
    "v2.4": ({"w_l10": 2.5, "w_prev": 0.2, "toi_recent": 0.8, "m_w_l10": 2.5},
             "Forme récente : les 10 derniers matchs pèsent beaucoup plus, la saison passée moins"),
    "v2.5": ({"k_pp": 1.8, "pen_exp": 1.4, "score_s": 0.22, "k_pp_passes": 2.4},
             "Situations spéciales : avantage numérique, indiscipline et état de match amplifiés"),
}
for _id, (_diff, _desc) in GEN2.items():
    VERSIONS[_id] = {"params": {**V2, **_diff}, "famille": "v2", "generation": 2,
                     "nb_k": V2["nb_k"], "desc": _desc}
# La principale reste v1 tant qu'une autre n'a pas prouvé mieux (voir auto.py)
# --- VA : scan complet des paramètres, fait SUR DEMANDE (pas chaque semaine) ---------
# va.1 (23 octobre) : descente sur tous les paramètres joueurs de la famille v2, sur les 23
# premiers soirs. Trois départs testés ; estimation honnête (optimisé sur 70 % des soirs,
# testé sur les 30 % jamais vus) : +0.10 % vs v1 sur les joueurs. Sur tout l'historique
# (optimiste, il l'a vu) : +0.37 %. Le modèle de MATCH n'est PAS optimisé : sur 110 matchs,
# le scan empirait de 3.4 % les matchs jamais vus (il apprenait le bruit par cœur).
# Ses paramètres ont vu jusqu'au 22 octobre : il est noté seulement à partir du 24.
VA = {
    "va.1": ({
    "away": 0.95,
    "b2b": 1.0,
    "cal_bu_a": 1.06,
    "cal_bu_b": 0.07,
    "cal_pa_a": 1.1,
    "cal_pa_b": 0.02,
    "cal_t_a": 0.98,
    "cal_t_b": -0.03,
    "k_pp": 0.0,
    "k_pp_passes": 1.1,
    "nb_k": 35.4492,
    "opp_reg_gp": 12.375,
    "pa_pseudo_min": 439.23,
    "pen_exp": 0.0,
    "pseudo_min": 97.2,
    "scale_buts": 1.01,
    "scale_passes": 0.96,
    "scale_tirs": 0.99,
    "scale_tirs_D": 0.99,
    "score_s": 0.0,
    "sh_pseudo_tirs": 176.0,
    "sv_reg_gp": 60,
    "toi_recent": 0.36,
    "vig_props": 1.0,
    "vs_reg_gp": 44.2368,
    "w_att": 0.0,
    "w_l10": 1.188,
    "w_mod_joueur": 0.7563,
    "w_prev": 0.56
    }, "2026-10-23", "2026-10-24",
        "Scan complet du 23 oct. (23 soirs) : ajouts v2 coupés (PP, tentatives, indiscipline, état de match à 0), "
        "forme récente moins pesante, dispersion élargie, petite recalibration. Matchs = v2 de base."),
}
for _id, (_diff, _cree, _juge, _desc) in VA.items():
    VERSIONS[_id] = {"params": {**V2, **_diff}, "famille": "v2", "generation": "A", "nb_k": _diff.get("nb_k", V2["nb_k"]),
                     "desc": _desc, "cree": _cree, "juge_depuis": _juge}

VERSION_PRINCIPALE = "v1"
MARCHES = ["tirs1", "tirs", "tirs3", "tirs4", "buts", "passes"]
# Distribution des tirs de chaque version ayant déjà prédit (None = Poisson).
# Inclut les versions auto retirées : leur historique doit pouvoir être complété.
DISPERSION = {"v0": None, "v1": V1["nb_k"], "v2": V2["nb_k"], **{v: V2["nb_k"] for v in GEN2},
              **{v: VERSIONS[v]["nb_k"] for v in VA}}


def charger_auto(chemin):
    """Ajoute au registre les versions créées par le tuner (data/versions_auto.json).

    Chaque version auto = V1 avec des paramètres différents, GELÉS à sa création.
    Les versions retirées ne prédisent plus mais leur historique reste.
    Retourne la version principale choisie (v1 par défaut)."""
    import json
    try:
        etat = json.loads(open(chemin, encoding="utf-8").read())
    except (FileNotFoundError, ValueError):
        return VERSION_PRINCIPALE
    for v in etat.get("versions", []):
        fam = v.get("famille", "v1")
        base = FAMILLES[fam]["P"]
        DISPERSION[v["id"]] = v["params"].get("nb_k", base["nb_k"])
        if v.get("retiree"):
            VERSIONS.pop(v["id"], None)   # au cas où elle était chargée avant d'être retirée
            continue
        P = {**base, **v["params"]}
        P["vs_cap"], P["m_h2h_cap"] = tuple(P["vs_cap"]), tuple(P["m_h2h_cap"])
        VERSIONS[v["id"]] = {"params": P, "auto": True, "famille": fam, "desc": v["desc"], "cree": v["cree"],
                             "nb_k": P["nb_k"], "generation": v.get("generation"),
                             "juge_depuis": v.get("juge_depuis"), "cat": v.get("cat")}
    # Versions faites à la main (génération 2) éliminées par l'évolution : elles ne
    # prédisent plus, mais leur historique reste dans les fichiers.
    for vid in etat.get("retirees", []):
        if vid in VERSIONS and VERSIONS[vid].get("generation"):
            VERSIONS.pop(vid)
    p = etat.get("principale", VERSION_PRINCIPALE)
    return p if p in VERSIONS else VERSION_PRINCIPALE
