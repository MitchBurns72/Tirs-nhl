"""
Cotes automatiques : SportsGameOdds (sportsgameodds.com) ou The Odds API
(the-odds-api.com), selon la clé mise dans le secret ODDS_API_KEY.

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
import re
import statistics
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
REGION = "us"

# Marché Odds API -> (notre marché, ligne "Over" qui veut dire notre seuil)
# 1+ but = Yes / Over 0.5 ; 1+ passe = Over 0.5
# Tirs : ligne None -> on lit TOUTES les lignes alternatives du même appel
# (0.5, 1.5, 2.5, 3.5 = 1+, 2+, 3+, 4+). Zéro crédit de plus.
TIRS_PAR_LIGNE = {0.5: "tirs1", 1.5: "tirs", 2.5: "tirs3", 3.5: "tirs4"}
# Lignes alternatives (plan payant pour tout prendre) : ligne « plus de » -> notre marché
PAR_LIGNE = {"tirs": TIRS_PAR_LIGNE,
             "buts": {0.5: "buts", 1.5: "buts2", 2.5: "buts3"},
             "passes": {0.5: "passes", 1.5: "passes2"},
             "pts": {0.5: "pts1", 1.5: "pts2"}}
PROPS = {
    "player_shots_on_goal_alternate": ("tirs", None),
    "player_shots_on_goal": ("tirs", None),
    "player_goal_scorer_anytime": ("buts", "oui"),
    "player_goals": ("buts", None),
    "player_goals_alternate": ("buts", None),
    "player_assists": ("passes", None),
    "player_assists_alternate": ("passes", None),
    "player_points": ("pts", None),
    "player_points_alternate": ("pts", None),
}

etat = {"restants": None, "utilises": None, "erreur": None, "source": None, "test": None}


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


def recuperer_theoddsapi(matchs, noms_equipes, marches_props):
    """Retourne (cotes_matchs, cotes_joueurs).
    cotes_matchs[gameId]   = {"home": prix, "away": prix, "p_home_marche": p sans marge,
                              "ligne": 6.5, "over": prix, "under": prix, "n": nb de books}
    cotes_joueurs[(gameId, nom_normalisé)][marché] = {"prix": médiane, "n": nb de books}
    """
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
                            if ligne == "oui":    # buteur : Yes / No, sans ligne
                                if o["name"] in ("No", "Under"):
                                    continue
                                cible = "buts"
                            else:
                                if o["name"] not in ("Over", "Yes"):
                                    continue
                                cible = PAR_LIGNE[marche].get(o.get("point"))
                                if not cible:
                                    continue
                            prix.setdefault((gid, norm(nom)), {}).setdefault(cible, []).append(o["price"])
                for k, d in prix.items():
                    cotes_j[k] = {mk: {"prix": _mediane(v), "n": len(v)} for mk, v in d.items()}
    except Exception as ex:  # noqa: BLE001
        # Les cotes sont un bonus : une panne de l'API (quota épuisé, clé
        # invalide) ne doit JAMAIS empêcher les prédictions d'être publiées.
        etat["erreur"] = str(ex)[:200]
    return cotes_m, cotes_j


# =================================================================================
# SportsGameOdds (sportsgameodds.com) : 2e fournisseur, plan gratuit « Amateur »
# =================================================================================
# POURQUOI un 2e fournisseur : si la clé The Odds API n'arrive pas, celle-ci peut
# la remplacer SANS toucher au workflow. Même secret ODDS_API_KEY : on essaie
# SportsGameOdds d'abord ; si la clé y est refusée (401/403), on essaie The Odds API.
#
# Plan gratuit : 2 500 « objets »/mois (1 match = 1 objet), 10 requêtes/minute.
# Un seul appel par calcul avec cotes (~12 matchs/soir) ≈ 400 objets/mois.
# Un appel donne TOUT : victoires (avec et sans prolongation), totaux, props joueurs.
#
# Format (docs.sportsgameodds.com) : event.odds = { oddID: {...} }, oddID =
# {statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}. Cotes en format
# américain (« -110 », « +150 »). Hockey : buts = statID « points »,
# tirs = « shots_onGoal », passes = « assists ». periodID « game » inclut
# prolongation + tirs de barrage ; « reg » = 60 min seulement.
# Les lignes alternatives (includeAltLines) ne sont pas décrites en détail dans
# les docs : le lecteur ci-dessous ramasse toute paire (overUnder, odds) qu'il
# trouve dans byBookmaker, peu importe où elle est rangée.

SGO_BASE = "https://api.sportsgameodds.com/v2"
SGO_STAT = {"shots_onGoal": "tirs", "points": "buts", "assists": "passes"}


class CleRefusee(Exception):
    pass


def americain_vers_decimal(x):
    """'-110' -> 1.909 ; '+150' -> 2.5 ; 'EVEN' -> 2.0. None si illisible."""
    try:
        t = str(x).strip().upper()
        if t in ("EVEN", "EV"):
            return 2.0
        v = float(t)
    except (TypeError, ValueError):
        return None
    if v >= 100:
        return round(1 + v / 100, 6)
    if v <= -100:
        return round(1 + 100 / -v, 6)
    return None


def _paires(entree):
    """Toutes les paires (ligne, cote décimale) d'un bookmaker : la ligne principale
    et les alternatives, où qu'elles soient rangées dans l'objet."""
    out = []
    def visiter(o):
        if isinstance(o, dict):
            if "odds" in o and o.get("available", True) is not False:
                dec = americain_vers_decimal(o.get("odds"))
                if dec:
                    ligne = o.get("overUnder")
                    try:
                        ligne = float(ligne) if ligne is not None else None
                    except ValueError:
                        ligne = None
                    out.append((ligne, dec))
            for v in o.values():
                if isinstance(v, (dict, list)):
                    visiter(v)
        elif isinstance(o, list):
            for v in o:
                visiter(v)
    visiter(entree)
    return out


def _sgo_get(params):
    cle = os.environ.get("ODDS_API_KEY")
    url = f"{SGO_BASE}/events?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": cle, "User-Agent": "tirs-nhl/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise CleRefusee() from e
        raise


def analyser_sgo(evenements, matchs, noms_equipes):
    """Transforme les événements SportsGameOdds en nos deux dictionnaires de cotes.
    Séparé de l'appel réseau pour pouvoir le tester avec de fausses réponses."""
    cotes_m, cotes_j = {}, {}
    for m in matchs:
        nh, na = norm(noms_equipes.get(m["home"], "")), norm(noms_equipes.get(m["away"], ""))
        ev = next((e for e in evenements
                   if nh and na and nh in norm(e.get("teams", {}).get("home", {}).get("names", {}).get("long", ""))
                   and na in norm(e.get("teams", {}).get("away", {}).get("names", {}).get("long", ""))), None)
        if not ev:
            continue
        gid = m["id"]
        joueurs = {pid: norm(j.get("name") or f"{j.get('firstName', '')} {j.get('lastName', '')}")
                   for pid, j in (ev.get("players") or {}).items()}
        # (catégorie, clé) -> liste de cotes décimales, une par bookmaker
        tas = {}
        for o in (ev.get("odds") or {}).values():
            stat, ent, per, bt, side = (o.get("statID"), o.get("statEntityID"), o.get("periodID"),
                                        o.get("betTypeID"), o.get("sideID"))
            livres = (o.get("byBookmaker") or {}).values()
            if stat == "points" and per == "game" and bt == "ml" and side in ("home", "away"):
                for b in livres:
                    for _, dec in _paires(b)[:1]:
                        tas.setdefault(("m", side), []).append(dec)
            elif stat == "points" and per == "reg" and bt == "ml3way" and side in ("home", "away", "draw"):
                for b in livres:
                    for _, dec in _paires(b)[:1]:
                        tas.setdefault(("m", {"home": "h3", "away": "a3", "draw": "x3"}[side]), []).append(dec)
            elif stat == "points" and per == "game" and bt == "ou" and ent not in ("home", "away") and ent not in joueurs:
                for b in livres:
                    for ligne, dec in _paires(b):
                        ligne = ligne if ligne is not None else float(o.get("bookOverUnder") or 0)
                        if ligne in (5.5, 6.5) and side in ("over", "under"):
                            tas.setdefault(("t", str(ligne), side), []).append(dec)
            elif ent in joueurs and stat in SGO_STAT and per == "game" and bt in ("ou", "yn") and side in ("over", "yes"):
                marche = SGO_STAT[stat]
                for b in livres:
                    for ligne, dec in _paires(b):
                        if ligne is None:
                            try:
                                ligne = float(o.get("bookOverUnder"))
                            except (TypeError, ValueError):
                                ligne = 0.5 if bt == "yn" else None
                        cible = TIRS_PAR_LIGNE.get(ligne) if marche == "tirs" else (marche if ligne == 0.5 else None)
                        if cible:
                            tas.setdefault(("j", joueurs[ent], cible), []).append(dec)
        c = {}
        if ("m", "home") in tas and ("m", "away") in tas:
            c["home"], c["away"] = _mediane(tas[("m", "home")]), _mediane(tas[("m", "away")])
            ih, ia = 1 / c["home"], 1 / c["away"]
            c["p_home_marche"] = round(ih / (ih + ia), 4)
        for k in ("h3", "a3", "x3"):
            if ("m", k) in tas:
                c[k] = _mediane(tas[("m", k)])
        tot = {}
        for ligne in ("5.5", "6.5"):
            o_, u_ = tas.get(("t", ligne, "over")), tas.get(("t", ligne, "under"))
            if o_ or u_:
                tot[ligne] = {"over": _mediane(o_ or []), "under": _mediane(u_ or [])}
        if tot:
            c["totaux"] = tot
        if c:
            c["n"] = max((len(v) for v in tas.values()), default=0)
            c["source"] = "SportsGameOdds"
            cotes_m[gid] = c
        for (cat, *reste), v in tas.items():
            if cat == "j":
                nom, marche = reste
                cotes_j.setdefault((gid, nom), {})[marche] = {"prix": _mediane(v), "n": len(v)}
    return cotes_m, cotes_j


def recuperer_sgo(matchs, noms_equipes):
    debut = min(m["start"] for m in matchs)
    params = {"leagueID": "NHL", "oddsAvailable": "true", "includeAltLines": "true", "limit": 50,
              # Fenêtre : les matchs de la soirée seulement (économise les objets du forfait)
              "startsAfter": debut[:10] + "T12:00:00Z",
              "startsBefore": (max(m["start"] for m in matchs)[:10]) + "T23:59:59Z"}
    rep = _sgo_get(params)
    evs = rep.get("data", rep) if isinstance(rep, dict) else rep
    etat["source"] = "SportsGameOdds"
    etat["utilises"] = (etat.get("utilises") or 0) + len(evs or [])
    return analyser_sgo(evs or [], matchs, noms_equipes)


def tester_cle():
    """Vérifie la clé quand aucun match du soir n'est ouvert (lancement manuel) :
    lit les matchs à venir sans rien enregistrer. The Odds API : 1 crédit (h2h).
    Résultat dans etat["test"] (nombre de matchs trouvés) ou etat["erreur"]."""
    cle = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not cle:
        return
    try:
        if re.fullmatch(r"[0-9a-f]{32}", cle):
            etat["source"] = "The Odds API"
            evs = _get("/odds", {"markets": "h2h"})
        else:
            etat["source"] = "SportsGameOdds"
            rep = _sgo_get({"leagueID": "NHL", "oddsAvailable": "true", "limit": 1})
            evs = rep.get("data", rep) if isinstance(rep, dict) else rep
        etat["test"] = len(evs or [])
    except Exception as ex:  # noqa: BLE001
        etat["erreur"] = ("clé refusée" if isinstance(ex, CleRefusee) else str(ex)[:200])


def recuperer(matchs, noms_equipes, marches_props):
    """Point d'entrée : choisit le fournisseur selon la clé.

    Une clé The Odds API = exactement 32 caractères hexadécimaux (0-9, a-f) : on va
    direct chez eux. Sinon on essaie SportsGameOdds, et à la MOINDRE erreur
    (clé refusée, réponse inattendue, panne), on retombe sur The Odds API."""
    cle = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not cle or not matchs:
        return {}, {}
    if re.fullmatch(r"[0-9a-f]{32}", cle):
        etat["source"] = "The Odds API"
        return recuperer_theoddsapi(matchs, noms_equipes, marches_props)
    try:
        return recuperer_sgo(matchs, noms_equipes)
    except Exception as ex:  # noqa: BLE001
        erreur_sgo = "clé refusée" if isinstance(ex, CleRefusee) else str(ex)[:120]
        etat["source"] = "The Odds API"
        etat["erreur"] = None
        res = recuperer_theoddsapi(matchs, noms_equipes, marches_props)
        if etat["erreur"]:
            etat["erreur"] = f"SportsGameOdds : {erreur_sgo} · The Odds API : {etat['erreur']}"
        return res
