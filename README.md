# La Patente à Tirs

Chaque soir de match NHL :
- **Joueurs** : probabilité de 2+ tirs au but, d'un but, d'une passe
- **Matchs** : % de victoire, buts attendus, total projeté, P(plus de 5.5 / 6.5)
- **Cotes** : médiane des bookies US en auto (The Odds API) + champ Mise-o-jeu manuel
- **Suivi** : note /100, calibration et bon côté par marché, pour chaque version de l'algo

## Mes bets

Onglet **Bets** : bouton « + Bet » sur chaque joueur, victoire et total (plus/moins 5.5 et 6.5).
Chaque bet est **Test** (sans argent) ou **Vrai bet**, avec stats séparées : profit, rendement,
réussite réelle vs réussite prévue par l'algo, détail par marché.
Statut live : jaune = en cours, vert = gagné (même avant la fin si c'est acquis), rouge = perdu,
gris = pas commencé ou annulé (joueur qui n'a pas joué = remboursé).
Les bets sont sauvegardés dans `data/bets.json` du repo (carte « Sauvegarde GitHub » : clé à accès
limité au repo, permission Contents en lecture/écriture), synchronisés entre appareils aux 2 minutes.
Un bet supprimé reste dans le fichier marqué `supprime` pour ne pas réapparaître. Les soirs passés sont réglés avec `docs/live/AAAA-MM-JJ.json`
(écrit par le calcul, gardé 21 jours) ; un bet réglé garde son résultat sur l'appareil.

## Labo (auto-upgrade)

`src/auto.py`, à chaque calcul :
- **Patterns** : segments où la version principale se trompe systématiquement (écart >= 2.5 écarts-types).
- **Tuner** (aux ~7 jours) : cherche de meilleurs paramètres sur les vieux soirs, valide sur les soirs récents
  jamais utilisés pour chercher. Si ça bat la principale d'au moins 0.5 % sans empirer un autre marché,
  crée une version gelée `aMMJJ` qui roule en parallèle. Max 3 versions auto actives.
- **Promotion** : une version auto devient principale seulement si elle bat la principale EN DIRECT,
  sur les mêmes prédictions, après 1500 prédictions de tirs.
- Une version auto ne prédit jamais un soir d'avant sa création (ses soirs d'entraînement).
- État : `data/versions_auto.json`. Pour annuler une version : y mettre `"retiree": true`.

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

Deux fournisseurs possibles, **un seul secret** `ODDS_API_KEY` : le code essaie SportsGameOdds,
et si la clé y est refusée, The Odds API.
- **SportsGameOdds** (plan gratuit Amateur, 2 500 matchs lus/mois) : victoires avec et sans prolongation,
  totaux 5.5 et 6.5, tirs 1+ à 4+, buts, passes. Un appel par calcul avec cotes.
- **The Odds API** (500 crédits/mois) : victoires, ligne de total principale, tirs.


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
