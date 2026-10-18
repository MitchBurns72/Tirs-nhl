"""
Couche d'accès à l'API publique de la NHL (api-web.nhle.com).

POURQUOI une couche séparée :
- Toute la logique réseau (retries, délais, cache mémoire) est à une seule place.
- Les tests peuvent remplacer `get_json` par des données bidon sans toucher au modèle.
- Si la NHL change une URL (ça arrive), on corrige ici et nulle part ailleurs.

On utilise seulement la librairie standard (urllib) : zéro dépendance à installer
dans la GitHub Action, donc rien qui peut casser à cause d'une mise à jour de package.
"""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://api-web.nhle.com/v1"

# Cache mémoire pour la durée d'une exécution : si deux équipes de la soirée
# demandent la même ressource, on ne frappe pas l'API deux fois.
_cache = {}


def get_json(path, retries=4):
    """GET sur l'API NHL avec retries et backoff exponentiel.

    POURQUOI le backoff : l'API publique n'a pas de quota documenté, mais elle
    peut répondre 429/5xx si on la martèle. On attend 1s, 2s, 4s... avant de
    réessayer plutôt que de planter toute l'exécution pour une erreur passagère.
    """
    if path in _cache:
        return _cache[path]
    url = f"{BASE}{path}"
    delai = 1.0
    for essai in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tirs-nhl/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
                _cache[path] = data
                return data
        except urllib.error.HTTPError as e:
            # 404 = la ressource n'existe pas (ex. joueur recrue sans saison précédente).
            # Inutile de réessayer, on retourne None et l'appelant gère l'absence.
            if e.code == 404:
                _cache[path] = None
                return None
            if essai == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if essai == retries - 1:
                raise
        time.sleep(delai)
        delai *= 2
    return None


def fetch_many(paths, workers=8):
    """Télécharge plusieurs ressources en parallèle.

    POURQUOI 8 threads : une soirée chargée = ~650 joueurs. Un par un, ça
    prendrait plusieurs minutes; à 8 en parallèle ça reste poli envers l'API
    tout en finissant en moins d'une minute.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        resultats = list(ex.map(get_json, paths))
    return dict(zip(paths, resultats))


# --- Raccourcis lisibles pour chaque endpoint utilisé ---------------------------

def schedule(date_iso):
    """Horaire de la semaine commençant à `date_iso` (on prend gameWeek[0])."""
    return get_json(f"/schedule/{date_iso}")


def roster(team):
    """Alignement actuel : forwards, defensemen, goalies."""
    return get_json(f"/roster/{team}/current")


def game_log_path(player_id, season):
    """Chemin du game log saison régulière (gameTypeId 2) d'un joueur."""
    return f"/player/{player_id}/game-log/{season}/2"


def club_stats(team, season=None):
    """Stats d'équipe; les gardiens donnent les tirs accordés (shotsAgainst)."""
    if season:
        return get_json(f"/club-stats/{team}/{season}/2")
    return get_json(f"/club-stats/{team}/now")


def standings_now():
    return get_json("/standings/now")


def club_schedule(team, season=None):
    """Tous les matchs d'une équipe avec les scores. Sert au modèle de match :
    buts pour/contre, 10 derniers, duels directs."""
    return get_json(f"/club-schedule-season/{team}/{season or 'now'}")


def boxscore(game_id):
    """Feuille de match : sert à grader les prédictions (champ `sog` par joueur)."""
    return get_json(f"/gamecenter/{game_id}/boxscore")
