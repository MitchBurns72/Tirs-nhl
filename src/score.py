"""
Tracking de réussite : est-ce que nos % veulent dire quelque chose?

On mesure trois affaires pour chaque version :

1. CALIBRATION (ton "pct de probabilité vs la réussite") : on regroupe les
   prédictions par tranche (70-80 %, 80-90 %...) et on compare le % prédit
   moyen au % réel. Un modèle bien calibré qui dit 80 % doit réussir ~80 % du temps.

2. BRIER SCORE : moyenne de (p - résultat)². 0 = parfait. C'est LA mesure
   standard pour des probabilités : elle punit autant l'excès de confiance
   que le manque de confiance.

3. NOTE GLOBALE /100, basée sur le Brier Skill Score (BSS) :
   BSS = 1 - Brier_modèle / Brier_référence
   La référence = dire le même % à tout le monde (le taux réel moyen de 2+ tirs).
   Note = 50 + 50 × BSS
     50  = pas mieux que deviner la moyenne pour tout le monde
     100 = parfait (impossible en sport)
     < 50 = pire que deviner la moyenne -> version à jeter
   Réalité du sport : un bon modèle tourne autour de 55-60. Un 58 c'est solide.
"""

import math

TRANCHES = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def resumer(fichiers, versions):
    """`fichiers` = liste de dicts de prédictions (un par date), déjà gradés en partie."""
    out = {}
    for v in versions:
        lignes = []      # (date, p, y)
        for f in fichiers:
            for e in f["entries"]:
                if e.get("result") is None or not e.get("played"):
                    continue  # pas encore joué, ou joueur scratché -> annulé
                pv = e["preds"].get(v)
                if pv is None:
                    continue  # version créée après cette date : pas de triche rétroactive
                lignes.append((f["date"], pv["p"], 1 if e["result"] >= 2 else 0))
        out[v] = _metriques(lignes)
    return out


def _metriques(lignes):
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
            calib.append({
                "tranche": f"{int(lo*100)}-{min(int(hi*100), 100)}",
                "n": len(t),
                "predit": sum(p for p, _ in t) / len(t),
                "reel": sum(y for _, y in t) / len(t),
            })

    # Top 10 de chaque soir : c'est ce que tu regardes en pratique sur ton cell.
    par_date = {}
    for d, p, y in lignes:
        par_date.setdefault(d, []).append((p, y))
    soirs = []
    for d in sorted(par_date):
        top = sorted(par_date[d], key=lambda t: -t[0])[:10]
        soirs.append({
            "date": d, "n": len(top),
            "hits": sum(y for _, y in top),
            "attendu": sum(p for p, _ in top),
        })
    top_n = sum(s["n"] for s in soirs)
    top_hits = sum(s["hits"] for s in soirs)
    top_att = sum(s["attendu"] for s in soirs)

    return {
        "n": n,
        "base": base,
        "brier": brier,
        "bss": bss,
        "note": max(0.0, min(100.0, 50 + 50 * bss)),
        "logloss": logloss,
        "calib": calib,
        "top10": {"n": top_n, "hits": top_hits, "attendu": top_att},
        "soirs": soirs[-14:],
    }
