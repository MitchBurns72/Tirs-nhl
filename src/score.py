"""
Tracking de réussite : est-ce que nos % veulent dire quelque chose?

Pour chaque version ET chaque marché (tirs 2+, but, passe, victoire, total) :

1. CALIBRATION (ton "pct de probabilité vs la réussite") : on regroupe les
   prédictions par tranche (70-80 %, 80-90 %...) et on compare le % prédit
   moyen au % réel. Un modèle bien calibré qui dit 80 % réussit ~80 % du temps.

2. BRIER SCORE : moyenne de (p - résultat)². 0 = parfait. C'est LA mesure
   standard pour des probabilités : elle punit autant l'excès de confiance
   que le manque de confiance.

3. NOTE /100, basée sur le Brier Skill Score (BSS) :
   BSS = 1 - Brier_modèle / Brier_référence
   La référence = dire le même % à tout le monde (le taux réel moyen).
   Note = 50 + 50 × BSS
     50  = pas mieux que deviner la moyenne pour tout le monde
     100 = parfait (impossible en sport)
     < 50 = pire que deviner la moyenne -> version à jeter
   En sport, 55-60 c'est un bon modèle. Pour les matchs (victoire), même les
   meilleurs modèles publics tournent autour de 52-55 : le hockey est très aléatoire.

4. NOTE GLOBALE d'une version : moyenne des notes de ses marchés, pondérée par
   le nombre de prédictions gradées de chacun.
"""

import math

TRANCHES = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
SEUILS = {"tirs": 2, "buts": 1, "passes": 1}
MARCHES_MATCH = {
    "victoire": lambda p: p["p_home"],
    "total55": lambda p: p["p_over"]["5.5"],
    "total65": lambda p: p["p_over"]["6.5"],
}


def p_joueur(e, v, m):
    """Proba d'un joueur. Compatible avec l'ancien format (tirs seulement : preds[v] = {p, lam})."""
    pv = e.get("preds", {}).get(v)
    if not pv:
        return None
    if m in pv:
        return pv[m]["p"]
    if m == "tirs" and "p" in pv:
        return pv["p"]
    return None


def r_joueur(e, m):
    r = e.get("result")
    if isinstance(r, dict):
        return r.get(m)
    return r if m == "tirs" else None   # ancien format : un nombre = tirs


def resumer(fichiers, versions):
    out = {"joueurs": {}, "matchs": {}, "global": {}}
    for m, seuil in SEUILS.items():
        out["joueurs"][m] = {}
        for v in versions:
            lignes = []
            for f in fichiers:
                for e in f["entries"]:
                    if not e.get("played"):
                        continue  # pas encore joué, ou scratché -> annulé
                    p, r = p_joueur(e, v, m), r_joueur(e, m)
                    if p is None or r is None:
                        continue  # version/marché créé après cette date : pas de triche rétroactive
                    lignes.append((f["date"], p, 1 if r >= seuil else 0))
            out["joueurs"][m][v] = _metriques(lignes, top=10)

    for m, get_p in MARCHES_MATCH.items():
        out["matchs"][m] = {}
        for v in versions:
            lignes, erreurs = [], []
            for f in fichiers:
                for g in f.get("games", []):
                    res, pv = g.get("result"), g.get("preds", {}).get(v)
                    if not res or not pv:
                        continue
                    h, a = res["home"], res["away"]
                    y = {"victoire": h > a, "total55": h + a > 5.5, "total65": h + a > 6.5}[m]
                    lignes.append((f["date"], get_p(pv), 1 if y else 0))
                    erreurs.append(abs(pv["total"] - (h + a)))
            s = _metriques(lignes, top=None)
            if erreurs and m == "total55":
                # Erreur moyenne sur le total de buts projeté (en buts)
                s["mae_total"] = sum(erreurs) / len(erreurs)
            out["matchs"][m][v] = s

    for v in versions:
        blocs = [out["joueurs"][m][v] for m in SEUILS] + [out["matchs"][m][v] for m in MARCHES_MATCH]
        n = sum(b["n"] for b in blocs if b["n"])
        out["global"][v] = {"n": n, "note": sum(b["note"] * b["n"] for b in blocs if b["n"]) / n if n else None}
    return out


def _metriques(lignes, top):
    n = len(lignes)
    if n == 0:
        return {"n": 0}
    base = sum(y for _, _, y in lignes) / n
    brier = sum((p - y) ** 2 for _, p, y in lignes) / n
    brier_ref = base * (1 - base) or 1e-9
    bss = 1 - brier / brier_ref
    eps = 1e-6
    logloss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                   for _, p, y in lignes) / n

    calib = []
    for lo, hi in TRANCHES:
        t = [(p, y) for _, p, y in lignes if lo <= p < hi]
        if t:
            calib.append({"tranche": f"{int(lo*100)}-{min(int(hi*100), 100)}", "n": len(t),
                          "predit": sum(p for p, _ in t) / len(t),
                          "reel": sum(y for _, y in t) / len(t)})

    # % de fois où le côté jugé favori (p >= 50 %) est arrivé. Lisible pour les matchs :
    # « le modèle a choisi le bon gagnant 58 % du temps ».
    precision = sum(1 for _, p, y in lignes if (p >= 0.5) == (y == 1)) / n
    out = {"n": n, "base": base, "brier": brier, "bss": bss, "precision": precision,
           "note": max(0.0, min(100.0, 50 + 50 * bss)), "logloss": logloss, "calib": calib}

    # Par soir : top 10 des joueurs (ce que tu regardes sur ton cell) ou tous les matchs.
    par_date = {}
    for d, p, y in lignes:
        par_date.setdefault(d, []).append((p, y))
    soirs = []
    for d in sorted(par_date):
        choix = sorted(par_date[d], key=lambda t: -t[0])
        if top:
            choix = choix[:top]
        soirs.append({"date": d, "n": len(choix), "hits": sum(y for _, y in choix),
                      "attendu": sum(p for p, _ in choix)})
    out["top"] = {"n": sum(s["n"] for s in soirs), "hits": sum(s["hits"] for s in soirs),
                  "attendu": sum(s["attendu"] for s in soirs)}
    out["soirs"] = soirs[-14:]
    return out
