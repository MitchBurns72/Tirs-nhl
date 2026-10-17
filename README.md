# La Patente à Tirs

Classe chaque soir les joueurs de la NHL selon leur probabilité de faire **2 tirs au but ou plus**, compare avec les cotes Mise-o-jeu, et garde le compte de la réussite de chaque version de l'algo.

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
