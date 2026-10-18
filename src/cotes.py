"""
Cotes automatiques via The Odds API (the-odds-api.com).

IMPORTANT : ce sont les cotes des bookies AMÉRICAINS (DraftKings, FanDuel, etc.),
pas Mise-o-jeu. Loto-Québec n'a pas d'API publique, et lire leur site avec un bot
serait fragile et probablement contre leurs conditions. On s'en sert donc comme
« prix du marché » : si ta proba bat la cote médiane US, l'edge est probablement
réel, et tu vas ensuite valider sur Mise-o-jeu (champ manuel sur la page).

BUDGET (plan gratuit = 500 crédits/mois) :
  coût d'un appel = nb de marchés × nb de régions
  - matchs (h2h + totals), tous les matchs en un appel ... 2 crédits
  - props joueurs : 1 appel PAR MATCH, × nb de marchés de props
Par défaut on prend seulement les tirs (1 crédit/match) au calcul de 17 h 45 :
~10 matchs × 30 jours + 60 ≈ 360 crédits/mois. Ça rentre.
Pour avoir aussi buts et passes en auto, il faut le plan payant : changer
ODDS_PROPS dans le workflow (voir README).

La clé n'est JAMAIS dans le code : elle vient du secret GitHub ODDS_API_KEY.
"""

import json
import os
import statistics
import unicodedata
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
REGION = "us"

# Marché Odds API -> (notre marché, ligne "Over" qui veut dire notre seuil)
# 1+ but = Yes / Over 0.5 ; 1+ passe = Over 0.5
# Tirs : ligne None -> on lit TOUTES les lignes alternatives du même appel
# (0.5, 1.5, 2.5, 3.5 = 1+, 2+, 3+, 4+). Zéro crédit de plus.
TIRS_PAR_LIGNE = {0.5: "tirs1", 1.5: "tirs", 2.5: "tirs3", 3.5: "tirs4"}
PROPS = {
    "player_shots_on_goal_alternate": ("tirs", None),
    "player_shots_on_goal": ("tirs", None),
    "player_goal_scorer_anytime": ("buts", None),
    "player_assists": ("passes", 0.5),
}

etat = {"restants": None, "utilises": None, "erreur": None}


def norm(nom):
    """'Juraj Slafkovský' -> 'juraj slafkovsky'. Les deux sources n'écrivent
    pas toujours les accents pareil, donc on compare sans accents ni ponctuation."""
    s = unicodedata.normalize("NFKD", nom or "").encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def _get(path, params):
    cle = os.environ.get("ODDS_API_KEY")
    params = {**params, "apiKey": cle, "regions": REGION, "oddsFormat": "decimal"}
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as r:
        # Le quota restant arrive dans les en-têtes : on l'affiche sur la page
        # pour que tu voies venir le mur avant de le frapper.
        etat["restants"] = r.headers.get("x-requests-remaining")
        etat["utilises"] = r.headers.get("x-requests-used")
        return json.loads(r.read().decode("utf-8"))


def _mediane(xs):
    return round(statistics.median(xs), 3) if xs else None


def _associer(evenements, matchs, noms_equipes):
    """Associe chaque événement Odds API à un match NHL par le nom d'équipe.
    POURQUOI le surnom ('Canadiens', 'Mammoth') : il est unique dans la ligue et
    présent dans les deux sources, peu importe 'Montreal' vs 'Montréal'."""
    out = {}
    for m in matchs:
        nh, na = norm(noms_equipes.get(m["home"], "")), norm(noms_equipes.get(m["away"], ""))
        for e in evenements:
            if nh and na and nh in norm(e["home_team"]) and na in norm(e["away_team"]):
                out[m["id"]] = e
                break
    return out


def recuperer(matchs, noms_equipes, marches_props):
    """Retourne (cotes_matchs, cotes_joueurs).
    cotes_matchs[gameId]   = {"home": prix, "away": prix, "p_home_marche": p sans marge,
                              "ligne": 6.5, "over": prix, "under": prix, "n": nb de books}
    cotes_joueurs[(gameId, nom_normalisé)][marché] = {"prix": médiane, "n": nb de books}
    """
    if not os.environ.get("ODDS_API_KEY") or not matchs:
        return {}, {}
    cotes_m, cotes_j = {}, {}
    try:
        evs = _get("/odds", {"markets": "h2h,totals"})
        assoc = _associer(evs, matchs, noms_equipes)
        for gid, e in assoc.items():
            ph, pa, lignes = [], [], {}
            for b in e.get("bookmakers", []):
                for mk in b.get("markets", []):
                    if mk["key"] == "h2h":
                        for o in mk["outcomes"]:
                            (ph if o["name"] == e["home_team"] else pa).append(o["price"])
                    elif mk["key"] == "totals":
                        for o in mk["outcomes"]:
                            d = lignes.setdefault(o.get("point"), {"Over": [], "Under": []})
                            d.setdefault(o["name"], []).append(o["price"])
            c = {"home": _mediane(ph), "away": _mediane(pa), "n": len(e.get("bookmakers", []))}
            if c["home"] and c["away"]:
                # Retirer la marge du bookie : on normalise les deux probas implicites
                ih, ia = 1 / c["home"], 1 / c["away"]
                c["p_home_marche"] = round(ih / (ih + ia), 4)
            if lignes:
                # La ligne offerte par le plus de books = la ligne principale
                ligne = max(lignes, key=lambda k: len(lignes[k]["Over"]))
                c.update({"ligne": ligne, "over": _mediane(lignes[ligne]["Over"]),
                          "under": _mediane(lignes[ligne]["Under"])})
            cotes_m[gid] = c

        if marches_props:
            for gid, e in assoc.items():
                ev = _get(f"/events/{e['id']}/odds", {"markets": ",".join(marches_props)})
                prix = {}
                for b in ev.get("bookmakers", []):
                    for mk in b.get("markets", []):
                        if mk["key"] not in PROPS:
                            continue
                        marche, ligne = PROPS[mk["key"]]
                        for o in mk["outcomes"]:
                            nom = o.get("description") or o.get("name")
                            oui = o["name"] in ("Over", "Yes") or marche == "buts" and o["name"] not in ("No", "Under")
                            if not oui:
                                continue
                            cible = marche
                            if marche == "tirs":
                                cible = TIRS_PAR_LIGNE.get(o.get("point"))
                                if not cible:
                                    continue
                            elif ligne is not None and o.get("point") != ligne:
                                continue
                            prix.setdefault((gid, norm(nom)), {}).setdefault(cible, []).append(o["price"])
                for k, d in prix.items():
                    cotes_j[k] = {mk: {"prix": _mediane(v), "n": len(v)} for mk, v in d.items()}
    except Exception as ex:  # noqa: BLE001
        # Les cotes sont un bonus : une panne de l'API (quota épuisé, clé
        # invalide) ne doit JAMAIS empêcher les prédictions d'être publiées.
        etat["erreur"] = str(ex)[:200]
    return cotes_m, cotes_j
