"""
Registre des versions de l'algorithme.

RÈGLE D'OR : une version publiée ne se modifie JAMAIS.
On veut comparer v0 vs v1 vs v2 sur les mêmes matchs, en tout temps. Si on
retouche v1 après coup, ses vieilles prédictions ne correspondent plus au code
et la comparaison devient de la bouillie. Pour essayer une idée : on copie la
dernière version, on lui donne un nouveau nom (ex. "v2") et on l'ajoute à VERSIONS.

Toutes les versions roulent chaque jour sur la même soirée ("shadow mode").
C'est ça qui rend la comparaison juste : même joueurs, mêmes matchs, mêmes soirs.

Chaque version reçoit le même dictionnaire `ctx` (construit dans run.py) et
retourne un dict : {"p": P(2+ tirs), "lam": tirs attendus}.
"""

import math


# --- Distributions ---------------------------------------------------------------

def p2plus_poisson(lam):
    """P(X >= 2) pour une Poisson : 1 - P(0) - P(1) = 1 - e^-λ(1 + λ)."""
    return 1.0 - math.exp(-lam) * (1.0 + lam)


def p2plus_negbin(lam, k):
    """P(X >= 2) pour une binomiale négative de moyenne λ et dispersion k.

    POURQUOI pas juste Poisson : la Poisson suppose variance = moyenne. Dans la
    vraie vie, un joueur n'a pas le même contexte chaque soir (temps de PP,
    blowout, changement de trio), donc ses tirs varient PLUS que ça.
    Variance NB = λ + λ²/k. Plus k est grand, plus on se rapproche de Poisson.
    Effet concret : la Poisson surestime un peu les P(2+) des gros tireurs.

    Formules : p = k/(k+λ) ; P(0) = p^k ; P(1) = k·p^k·(1-p)
    """
    p = k / (k + lam)
    p0 = p ** k
    p1 = k * (p ** k) * (1.0 - p)
    return 1.0 - p0 - p1


# --- v0 : la baseline naïve ------------------------------------------------------

def v0(ctx):
    """Moyenne de tirs par match de la saison + Poisson. Rien d'autre.

    POURQUOI garder une version aussi simple : c'est l'étalon. Si v1 (avec
    adversaire, temps de glace, forme récente...) ne bat pas v0 sur des
    centaines de prédictions, c'est que toute la complexité ne sert à rien.
    """
    cur = ctx["cur"]
    if len(cur) >= 3:
        lam = sum(g["shots"] for g in cur) / len(cur)
    elif ctx["prev"]["gp"] > 0:
        # Début de saison : pas assez de matchs, on prend la saison passée.
        lam = ctx["prev"]["shots"] / ctx["prev"]["gp"]
    else:
        # Recrue sans historique : moyenne de sa position dans la ligue.
        lam = ctx["league"]["pos_rate"][ctx["pos"]] * ctx["league"]["pos_toi"][ctx["pos"]]
    lam = max(lam, 0.05)
    return {"p": p2plus_poisson(lam), "lam": lam}


# --- v1 : le modèle complet ------------------------------------------------------

V1 = {
    # Poids dans le taux de tirs "poolé" (tirs pondérés / minutes pondérées).
    # Les 10 derniers matchs comptent plus : la forme et le rôle actuel (trio, PP)
    # bougent pendant une saison.
    "w_l10": 1.5,
    "w_reste": 1.0,
    # La saison passée sert d'ancre en début de saison, mais on la plafonne
    # pour qu'elle n'enterre pas ce qui se passe cette année.
    "w_prev": 0.35,
    "prev_toi_cap": 900.0,      # minutes max de la saison passée considérées
    # Pseudo-minutes à la moyenne de la position : tire les petits échantillons
    # vers la moyenne (un 4e trio qui a fait 4 tirs en 2 matchs n'est pas un sniper).
    "pseudo_min": 120.0,
    # Temps de glace projeté : 60 % les 5 derniers matchs, 40 % la saison.
    "toi_recent": 0.6,
    # Régression des tirs accordés par l'adversaire : équivalent de 15 matchs
    # de saison passée. Après 15 matchs cette année, c'est 50/50.
    "opp_reg_gp": 15.0,
    "home": 1.03,
    "away": 0.97,
    "b2b": 0.96,               # 2e match en 2 soirs : un peu moins de jus/de temps
    # Historique contre l'adversaire : très bruité (3-4 matchs par an), donc
    # on le shrink fort vers 1.0 avec l'équivalent de 12 matchs neutres.
    "vs_reg_gp": 12.0,
    "vs_cap": (0.85, 1.15),
    "nb_k": 15.0,
}


def taux_poole(ctx, P):
    """Taux de tirs par minute combinant L10, reste de la saison, saison passée
    et une ancre à la moyenne de la position."""
    cur = ctx["cur"]
    l10 = cur[-10:]
    reste = cur[:-10]
    pos_rate = ctx["league"]["pos_rate"][ctx["pos"]]

    prev = ctx["prev"]
    prev_toi = min(prev["toi"], P["prev_toi_cap"])
    # On garde le TAUX de la saison passée, appliqué aux minutes plafonnées.
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
        # 1-2 matchs : on mélange avec la saison passée pour pas surréagir.
        moy_cur = sum(g["toi"] for g in cur) / len(cur)
        if ctx["prev"]["gp"] > 0:
            moy_prev = ctx["prev"]["toi"] / ctx["prev"]["gp"]
            return 0.5 * moy_cur + 0.5 * moy_prev
        return moy_cur
    if ctx["prev"]["gp"] > 0:
        return ctx["prev"]["toi"] / ctx["prev"]["gp"]
    return ctx["league"]["pos_toi"][ctx["pos"]]


def facteur_adversaire(ctx, P):
    """Tirs accordés par match de l'adversaire vs la moyenne de la ligue.
    > 1 = équipe passoire, < 1 = équipe qui bloque/contrôle la rondelle."""
    o = ctx["opp"]
    sa = (o["gp"] * o["sa_cur"] + P["opp_reg_gp"] * o["sa_prev"]) / (o["gp"] + P["opp_reg_gp"])
    return sa / ctx["league"]["sa_avg"]


def facteur_vs(ctx, P):
    """Ce que tu demandais : son historique contre CETTE équipe.
    Gardé, mais avec un shrink fort parce que 3-4 matchs = du bruit."""
    vs = ctx["vs"]
    if vs["toi"] <= 0 or vs["gp"] == 0 or ctx["base_rate"] <= 0:
        return 1.0
    ratio = (vs["shots"] / vs["toi"]) / ctx["base_rate"]
    adj = (vs["gp"] * ratio + P["vs_reg_gp"]) / (vs["gp"] + P["vs_reg_gp"])
    lo, hi = P["vs_cap"]
    return min(max(adj, lo), hi)


def v1(ctx):
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
        # Le détail sert à la page : tu vois POURQUOI un joueur est en haut.
        "detail": {
            "rate60": rate * 60, "toi": toi, "f_opp": f_opp,
            "f_lieu": f_lieu, "f_b2b": f_b2b, "f_vs": f_vs,
        },
    }


# --- Registre --------------------------------------------------------------------
# Ordre = ordre d'affichage. La DERNIÈRE est la version "principale" sur la page.
VERSIONS = {
    "v0": {"fn": v0, "desc": "Baseline : moyenne tirs/match + Poisson"},
    "v1": {"fn": v1, "desc": "Forme L10, temps de glace, adversaire, domicile, back-to-back, historique vs équipe"},
}
VERSION_PRINCIPALE = list(VERSIONS)[-1]
