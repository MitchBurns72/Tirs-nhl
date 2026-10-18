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
from datetime import date, datetime, timezone

from modeles import V1, modele_joueur_param, modele_match_param

# --- Seuils (les tests les baissent pour rouler sur une petite fausse ligue) ------
MIN_JOUEURS = 2000          # prédictions de joueurs gradées avec features avant d'oser tuner
MIN_MATCHS = 120            # idem pour les matchs
MIN_VAL_JOUEURS = 600       # taille minimale du jeu de validation (par marché joueur)
MIN_VAL_MATCHS = 40
GAIN_MIN = 0.005            # amélioration relative du Brier exigée en validation (0.5 %)
PIRE_MAX = 0.003            # dégradation tolérée sur un autre marché (0.3 %)
JOURS_ENTRE_TUNING = 6
MAX_AUTO_ACTIVES = 3
PROMO_MIN_TIRS = 1500       # prédictions EN DIRECT communes avant de promouvoir
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
SEUILS = {"tirs": 2, "buts": 1, "passes": 1}
NOMS = {"tirs": "Tirs 2+", "buts": "But", "passes": "Passe", "match": "Matchs"}


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


def _brier_joueurs(rows, P, marche):
    s = n = 0
    for _, f, r, _ in rows:
        if marche not in r:
            continue
        p = modele_joueur_param(f, P)[marche]["p"]
        y = 1 if r[marche] >= SEUILS[marche] else 0
        s += (p - y) ** 2; n += 1
    return s / n if n else None


def _brier_matchs(rows, P):
    """Somme des Brier victoire + total 5.5 (les deux décisions qu'on regarde)."""
    s = n = 0
    for _, f, (h, a), _ in rows:
        r = modele_match_param(f, P)
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
    for _ in range(passes):
        ameliore = False
        for nom, lo, hi, pas in ESPACE[groupe]:
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
# 3. Cycle complet (appelé par run.py à chaque calcul)
# =================================================================================

def _brier_commun(J, M, va, vb):
    """Brier moyen de va et vb sur les MÊMES prédictions en direct (pas le rétroactif,
    pas les soirs d'avant la naissance de la version)."""
    s = {va: 0.0, vb: 0.0}
    n = n_tirs = 0
    for _, _, r, preds in J:
        if va in preds and vb in preds:
            for m in SEUILS:
                if m in r and m in preds[va] and m in preds[vb]:
                    y = 1 if r[m] >= SEUILS[m] else 0
                    s[va] += (preds[va][m]["p"] - y) ** 2
                    s[vb] += (preds[vb][m]["p"] - y) ** 2
                    n += 1
                    n_tirs += m == "tirs"
    return (s[va] / n, s[vb] / n, n_tirs) if n else (None, None, 0)


def cycle(historique, chemin_etat, versions_actives, aujourdhui=None, force=False):
    """Patterns + tuning (si dû) + promotion. Met à jour data/versions_auto.json.
    Retourne l'état (pour la page) et True si une nouvelle version a été créée."""
    aujourdhui = aujourdhui or date.today().isoformat()
    try:
        etat = json.loads(open(chemin_etat, encoding="utf-8").read())
    except (FileNotFoundError, ValueError):
        etat = {"versions": [], "journal": [], "principale": "v1"}
    principale = etat.get("principale", "v1")
    if principale not in versions_actives:
        principale = "v1"
    J, M = donnees(historique)
    # Le rétroactif compte pour le tuning et les patterns (il est honnête : données d'avant),
    # mais la PROMOTION se juge seulement en direct, voir _brier_commun.
    etat["patterns"] = patterns(J, M, principale)
    etat["donnees"] = {"joueurs": len(J), "matchs": len(M)}
    nouvelle = False

    # --- Tuning ---
    derniere = etat.get("dernier_tuning")
    jours = (date.fromisoformat(aujourdhui) - date.fromisoformat(derniere)).days if derniere else 999
    if force or jours >= JOURS_ENTRE_TUNING:
        if len(J) < MIN_JOUEURS and len(M) < MIN_MATCHS:
            msg = f"Pas encore assez de données ({len(J)} joueurs / {len(M)} matchs gradés avec features)."
            P = None
        else:
            ref = next((v["params"] for v in etat["versions"] if v["id"] == principale), {})
            P, msg = tuner(J, M, {**V1, **ref}, aujourdhui)
        etat["dernier_tuning"] = aujourdhui
        if P:
            vid = "a" + aujourdhui[5:7] + aujourdhui[8:10]
            diff = {k: v for k, v in P.items() if V1.get(k, 1.0 if k.startswith("scale") else None) != v}
            etat["versions"].append({
                "id": vid, "cree": aujourdhui, "base": principale,
                "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in diff.items()},
                "desc": f"Auto du {aujourdhui} (base {principale}) : {msg}", "retiree": False})
            nouvelle = True
            msg = f"Nouvelle version {vid} créée. {msg}"
            # Max 3 versions auto actives : on retire la plus vieille qui n'est pas la principale.
            actives = [v for v in etat["versions"] if not v.get("retiree")]
            while len(actives) > MAX_AUTO_ACTIVES:
                vieille = next(v for v in actives if v["id"] != principale)
                vieille["retiree"] = True
                actives = [v for v in etat["versions"] if not v.get("retiree")]
        etat["journal"] = ([{"date": aujourdhui, "msg": msg}] + etat.get("journal", []))[:20]

    # --- Promotion ---
    for v in etat["versions"]:
        if v.get("retiree") or v["id"] == principale or v["id"] not in versions_actives:
            continue
        b_princ, b_auto, n_tirs = _brier_commun(J, M, principale, v["id"])
        v["essai"] = {"n_tirs": n_tirs, "gain": (b_princ - b_auto) / b_princ if b_princ else None}
        if n_tirs >= PROMO_MIN_TIRS and b_princ and (b_princ - b_auto) / b_princ >= PROMO_GAIN:
            etat["journal"].insert(0, {"date": aujourdhui,
                                       "msg": f"{v['id']} devient la version principale "
                                              f"({(b_princ - b_auto) / b_princ * 100:.1f} % mieux que {principale} en direct, {n_tirs} prédictions de tirs)."})
            principale = v["id"]
    etat["principale"] = principale
    etat["maj"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(chemin_etat, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(etat, ensure_ascii=False, indent=1))
    return etat, nouvelle
