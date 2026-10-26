# La Patente à Tirs

Chaque soir de match NHL :
- **Joueurs** : probabilité de 1+ à 4+ tirs au but, d'un but, d'une passe
- **Matchs** : % de victoire, buts attendus, total projeté, P(plus de 5.5 / 6.5)
- **Cotes** : médiane des bookies US en auto (The Odds API) + champ Mise-o-jeu manuel
- **Suivi** : note /100, calibration et bon côté par marché, pour chaque version de l'algo, et le Labo

## Ingrédients v2 (les v2.1 à v2.5 sont retirées, leurs ingrédients servent au scan)

La v2 = v1 + tentatives de tirs (cadrés + ratés + bloqués), temps en avantage numérique, indiscipline
de l'adversaire (minutes de PP qu'il donne), état de match (l'équipe qui tire de l'arrière tire plus),
matchs par les tirs (mélangés avec le modèle par buts), calibration de la victoire, et mélange avec
les cotes US (en logit) quand on les a. Données : `api.nhle.com/stats/rest` (rapports `realtime` et
`timeonice` par match, en cache dans `data/cache/`).

Cinq réglages roulent en même temps (`GEN2` dans `src/modeles.py`) :

| Version | Idée |
|---|---|
| v2.1 | Référence (mêmes paramètres que l'ancienne v2) |
| v2.2 | Fait plus confiance au marché (65-70 % du poids aux cotes US) |
| v2.3 | Fait plus confiance au modèle et aux tentatives de tirs |
| v2.4 | Forme récente : les 10 derniers matchs pèsent plus, la saison passée moins |
| v2.5 | Situations spéciales amplifiées (PP, indiscipline, état de match) |

Réglées AVANT de voir les résultats, donc rejouées honnêtement sur toute la saison (sans réseau :
les features sont déjà dans les fichiers). « Pourquoi ce % » montre le détail de chaque version,
dont le % du modèle seul, le % du marché et le mélange des deux.

## Matchs : alignements et projection joueur par joueur

`src/alignements.py` lit DailyFaceoff (trios, paires, PP1/PP2, PK, blessés, gardiens partants
avec leur statut) au calcul, puis le job live les rafraîchit aux 30 min dans `docs/live.json`
(midi à minuit). La page prend la version la plus récente.
Dans chaque match, avec la version principale :
- **Tirs les plus probables** par équipe : somme des tirs attendus des joueurs de l'alignement,
  dispersion binomiale négative (variance λ + λ²/15 par joueur), le mode + la plage de 80 %.
- **Score le plus probable** : somme des buts attendus des joueurs -> Poisson par équipe ->
  la case la plus probable du tableau des scores, plus les 2 suivantes et la victoire.
- Un joueur de l'alignement sans prédiction reçoit la moyenne de sa position dans son équipe.
- Limite : joueurs traités comme indépendants, donc les plages sont un peu étroites.

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

## Une recette par catégorie, recalculée chaque semaine

8 catégories, chacune avec SA recette : tirs 1+, tirs 2+, tirs 3+, tirs 4+, but, passe, victoire,
totaux 5.5 / 6.5. Pas de note globale, pas de concours de versions : le but est le meilleur % de
réussite possible dans chaque catégorie, avec toutes les données de la saison.

`src/auto.py`, une fois par semaine (au premier calcul 7 jours après le dernier) :
1. Pour chaque catégorie, scan de tous ses paramètres (descente par coordonnées) qui minimise son
   erreur (Brier), sur toute la saison.
2. Test honnête : même scan sur les 70 % plus vieux soirs, mesuré sur les 30 % plus récents jamais
   vus. La nouvelle recette remplace celle en service seulement si elle fait au moins 0.1 % mieux
   sur ces soirs-là (sinon on se mentirait : le scan apprend aussi le bruit de ce qu'il voit).
3. Gagnante : `tirs2_v5`, `but_v6`... Elle ne prédit que SON marché, est rejouée sur la saison
   pour l'affichage, et notée au Suivi seulement à partir de sa création.
Les nouvelles idées (ingrédients, autres façons de voir) s'ajoutent à la main quand on en trouve.
Ingrédients déjà disponibles pour le scan : tentatives, PP, indiscipline, état de match, patinoire,
prior selon le rôle, recalibration, mélange avec le marché, dispersion propre à chaque joueur.

**combo** (version par défaut) : chaque marché prend le % de la recette en service de sa catégorie.
Garde-fou : 1+ >= 2+ >= 3+ >= 4+ toujours (chaque seuil a sa recette).
**soir** : combo ajusté à l'alignement DailyFaceoff (temps de glace selon le trio, PP1/PP2,
gardien adverse), poids fixés d'avance, noté au Suivi comme les autres. Étiquette « Avant
l'alignement » / « Alignement confirmé ».
Patterns : segments où la recette des tirs 2+ se trompe systématiquement (z >= 2.5).

## Machine learning (`{cat}_ml`)

But : trouver tout seul les ratios et les combinaisons de données qui donnent le meilleur %, sans qu'on les devine à la main.

- **Données** : saison en cours + 3 saisons passées (box scores joueur par joueur, temps de glace, mises en échec, punitions, tirs bloqués, gardiens partants, calendrier, voyagement, fuseaux, météo extérieure à l'aréna). Récoltées petit à petit dans `data/hist/` (environ 8 min par calcul) : la saison en cours d'abord, puis une saison passée à la fois. Ça prend quelques calculs avant que tout soit là.
- **Features** : 50 par joueur (forme 5/10/20 matchs, saison, saison passée, TOI, par 60, tentatives, PP, % de tir, régularité, repos, back-to-back, km, fuseau, défense/gardien/physique adverse, historique vs l'équipe, aréna, météo, mois) et 29 par match. Toujours calculées avec ce qui était connu AVANT le match (pas de triche).
- **Modèle** : gradient boosting (des milliers d'arbres de décision qui testent eux-mêmes les seuils et les interactions entre données). Un modèle par catégorie (tirs 1+ à 4+, but, 2+/3+ buts, passe, 2+ passes, 1+/2+ pts, victoire, total 5.5/6.5).
- **Honnête** : les soirs passés sont prédits semaine par semaine avec seulement ce qui existait avant (walk-forward). Les versions `tirs2_ml`, `but_ml`, etc. apparaissent dans les puces et dans le Suivi de leur catégorie.
- **Chercheur de ratios** : toutes les paires de données en a÷b, a×b, a−b (3675 expressions joueur), puis les 40 meilleures recombinées avec chaque donnée (~6000 de plus, ex. (tirs/60 ÷ % arrêts adv.) × TOI). Chaque expression est notée « à proba égale » : est-ce qu'elle sépare ceux qui ont réussi de ceux qui ont manqué parmi ceux à qui le modèle donnait la même chance, mesuré hors échantillon (appris sur les semaines paires, vérifié sur les impaires et vice versa) pour que les chanceuses ne passent pas. Les meilleures sont ajoutées au modèle une à une si elles aident, puis gardées seulement si le modèle fait mieux avec sur les soirs jamais vus. Pas de log/racine/carré : un arbre coupe à un seuil, donc ça ne testerait rien de nouveau. Testé : retrouve un ratio caché à 3 données (x3 × x7 ÷ x9) parmi 50 données et 190 000 lignes. Refait chaque semaine et quand une saison passée s'ajoute ; max 6 min par calcul, la recherche continue au calcul suivant là où elle était (`data/hist/ratios_encours.json`, sélection en service dans `data/hist/ratios.json`). Résultats dans le Labo.
- **Juge hebdo** : le ML prend une catégorie seulement s'il bat la recette en service sur les 30 % de soirs les plus récents qu'il n'a jamais vus (minimum 1000 prédictions et 20 réussites ; 3+ buts est souvent trop rare pour juger). Sinon la recette reste.
- **Limites** : pas d'historique gratuit des alignements DailyFaceoff ni des cotes, donc le ML ne les voit pas. Les données qui ne servent à rien (la météo, probablement) finissent juste avec un poids de zéro.
- Fermer le ML : variable `ML=0`.

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

1. Dans `src/modeles.py`, ajoute une ligne dans `GEN2` (ou une nouvelle famille) avec les paramètres à changer.
2. **Ne modifie jamais une version déjà publiée.** Ses vieilles prédictions ne correspondraient plus au code et la comparaison serait faussée.
3. `python tests/test_simulation.py` pour vérifier que rien ne plante, puis commit.

Une nouvelle version commence à zéro prédiction gradée. Elle est comparée aux autres seulement sur les soirs où elle a roulé. Attends quelques centaines de prédictions avant de conclure.

## Installation (une seule fois)

1. Repo **public** sur GitHub (Pages gratuit l'exige; il n'y a rien de privé dedans, tes cotes restent sur ton cell).
2. Settings → Pages → Source : *Deploy from a branch*, branche `main`, dossier `/docs`.
3. Settings → Actions → General → Workflow permissions : *Read and write*.
4. Actions → *Calcul quotidien* → *Run workflow* pour le premier calcul.
5. Ouvre `https://<ton-user>.github.io/<nom-du-repo>/` sur ton cell et ajoute-la à l'écran d'accueil.
