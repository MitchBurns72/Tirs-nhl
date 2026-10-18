# La Patente à Tirs

Chaque soir de match NHL :
- **Joueurs** : probabilité de 2+ tirs au but, d'un but, d'une passe
- **Matchs** : % de victoire, buts attendus, total projeté, P(plus de 5.5 / 6.5)
- **Cotes** : médiane des bookies US en auto (The Odds API) + champ Mise-o-jeu manuel
- **Suivi** : note /100, calibration et bon côté par marché, pour chaque version de l'algo

## Live

Bouton **LIVE** en haut de la page : rafraîchit aux 30 secondes.
Vert = objectif atteint, jaune = match en cours pas encore atteint, rouge = match fini, raté.
Source : l'API NHL directement si ton navigateur l'accepte (temps réel), sinon `docs/live.json`,
mis à jour par GitHub aux ~5 minutes pendant les matchs (`src/live.py`). GitHub peut retarder
ses crons de quelques minutes les soirs chargés.

## Rétroactif

À chaque calcul, les soirs de la saison qui n'ont pas de prédictions sont rejoués avec seulement
les données connues avant ce soir-là (game logs, calendriers et feuilles de match datés), puis gradés.
Seule limite : l'alignement utilisé est celui d'aujourd'hui. Fichiers marqués `"retro": true`.

## Cotes automatiques

1. Compte gratuit sur the-odds-api.com, la clé arrive par courriel.
2. Repo → Settings → Secrets and variables → Actions → New repository secret : nom `ODDS_API_KEY`, valeur = ta clé.
3. Les cotes sont prises au calcul de 17 h 45 et aux lancements manuels seulement (budget de 500 crédits/mois).
4. Par défaut : matchs + tirs. Pour buts et passes en auto, il faut le plan payant ; change `ODDS_PROPS` dans le workflow.

Ce sont les cotes US (DraftKings, FanDuel...), pas Mise-o-jeu : Loto-Québec n'a pas d'API publique.

## Comment ça marche

Une GitHub Action roule `src/run.py` trois fois par jour (10 h, 14 h, 17 h 45, heure de l'Est) :

1. **Grade** les soirs passés : chaque prédiction reçoit les tirs réels de la feuille de match. Un joueur scratché est annulé, pas compté comme un échec.
2. **Prédit** les matchs de ce soir pas encore commencés, avec **toutes** les versions de l'algo. Un match commencé est gelé.
3. **Publie** `docs/data.json`. La page `docs/index.html` le lit.

Données : API publique de la NHL (`api-web.nhle.com`), gratuite, sans clé.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `src/modeles.py` | Les versions de l'algo (v0, v1, ...) |
| `src/run.py` | Pipeline : grader, prédire, publier |
| `src/score.py` | Note /100, Brier, calibration, top 10 |
| `src/nhl_api.py` | Appels à l'API NHL avec retries |
| `data/predictions/AAAA-MM-JJ.json` | Toutes les prédictions et leurs résultats |
| `data/cache/` | Saison passée résumée (ne change plus) |
| `tests/test_simulation.py` | Test complet sur une fausse ligue, sans internet |

## Ajouter une version de l'algo

1. Dans `src/modeles.py`, copie `v1` et ses paramètres `V1`, renomme en `v2` / `V2`, change ce que tu veux tester.
2. Ajoute `"v2"` à la fin de `VERSIONS`. La dernière de la liste devient la version principale de la page.
3. **Ne modifie jamais une version déjà publiée.** Ses vieilles prédictions ne correspondraient plus au code et la comparaison serait faussée.
4. `python tests/test_simulation.py` pour vérifier que rien ne plante, puis commit.

Une nouvelle version commence à zéro prédiction gradée. Elle est comparée aux autres seulement sur les soirs où elle a roulé. Attends quelques centaines de prédictions avant de conclure.

## Installation (une seule fois)

1. Repo **public** sur GitHub (Pages gratuit l'exige; il n'y a rien de privé dedans, tes cotes restent sur ton cell).
2. Settings → Pages → Source : *Deploy from a branch*, branche `main`, dossier `/docs`.
3. Settings → Actions → General → Workflow permissions : *Read and write*.
4. Actions → *Calcul quotidien* → *Run workflow* pour le premier calcul.
5. Ouvre `https://<ton-user>.github.io/<nom-du-repo>/` sur ton cell et ajoute-la à l'écran d'accueil.
