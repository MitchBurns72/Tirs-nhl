# La Patente à Tirs

Chaque soir de match NHL :
- **Joueurs** : probabilité de 1+ à 4+ tirs au but, d'un but, d'une passe
- **Matchs** : % de victoire, buts attendus, total projeté, P(plus de 5.5 / 6.5)
- **Cotes** : médiane des bookies US en auto (The Odds API) + champ Mise-o-jeu manuel
- **Suivi** : note /100, calibration et bon côté par marché, pour chaque version de l'algo, et le Labo

## v2 (génération 2 : v2.1 à v2.5)

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

## Champions par catégorie et version « combo »

La meilleure recette pour les tirs n'est pas forcément la meilleure pour les passes ou les matchs.
Chaque catégorie (tirs 1+ à 4+, but, passe, victoire, totaux) a son champion. **combo**, la version par
défaut de la page, prend le % de chaque marché chez le champion de sa catégorie (copié tel quel, rien
n'est recalculé ; « Pourquoi ce % » dit d'où il vient). Toutes les versions actives sont candidates.
Un champion change seulement si c'est prouvé : au moins 0.2 % d'erreur en moins, z >= 2.5 (plus
sévère, on compare beaucoup de versions dans 5 catégories), assez de lignes, et pour une version auto
ou VA, tenir aussi en direct. Les champions ne sont jamais retirés par l'évolution.
La reproduction suit la même logique : les enfants partent d'une recette combinée (paramètres tirs du
meilleur aux tirs, but du meilleur au but, passe du meilleur à la passe, match du meilleur aux matchs).

## Labo : évolution par générations

`src/auto.py`, à chaque calcul :
- **Classement** : chaque membre de la génération est comparé à v1 sur EXACTEMENT les mêmes joueurs
  (Brier des 6 marchés joueur) et les mêmes matchs. z = solidité, calculé par soir (les joueurs d'un
  même soir ne sont pas indépendants, donc c'est plus sévère qu'un z ligne par ligne).
- **Promotion** : le meneur devient la version principale (celle par défaut sur la page) seulement
  s'il bat la principale actuelle d'au moins 0.5 %, avec z >= 2.33 (moins de 1 % de chance que ce soit
  de la chance, ce qui compense le fait qu'on compare 5 candidats), sur au moins 1500 joueurs, sans
  être clairement pire sur les matchs.
- **Une génération par semaine** : 7 jours après sa naissance (et au moins 3 soirs en direct pour
  chaque nouvelle version), tout le monde est analysé avec la semaine de nouvelles données. Le meilleur
  survit (le parent, même si son avance n'est pas encore prouvée : il faut explorer), et 5 enfants
  aux personnalités différentes naissent. Une version trop jeune (moins de 500 joueurs notés) passe à
  la génération suivante au lieu d'être éliminée :
  1. **Tuner** : raffine le parent (cherche sur les vieux soirs, valide sur les récents).
  2. **Calibreur** : corrige les biais d'ensemble (échelle, dispersion, recalibration en logit).
  3. **Expérience** : si toute la génération perd contre v1, coupe les ajouts v2 à 40 % pour voir si
     ce sont eux qui nuisent ; sinon croisement avec le 2e (s'il n'est pas un clone).
  4. **Exploratrice** : grosse mutation (5 paramètres joueurs + 2 match).
  5. **Radicale** : 8 paramètres tirés n'importe où dans leurs bornes, surtout côté tirs.
  Chaque enfant doit donner des % vraiment différents du parent ET de ses frères (écart mesuré sur
  les % du modèle seul) ; sinon la mutation est refaite plus fort. Croisement : les paramètres de
  match viennent du meilleur sur les matchs. Les perdants arrêtent de prédire, leur historique reste.
- **Nouveaux ingrédients** (neutres au départ, l'évolution décide) : biais de patinoire (tirs comptés
  à domicile ÷ à l'étranger pour chaque aréna, régressé), prior selon le rôle (le taux de départ
  dépend du temps de glace), recalibration tirs / but / passe.
- **z** : affiché seulement après 7 soirs (avant, il ne veut rien dire).
- **VA (scan complet, sur demande)** : descente sur TOUS les paramètres joueurs, sur tout l'historique,
  avec plusieurs points de départ. Rejoint le concours de la semaine, notée seulement après sa date de
  scan (elle a tout vu avant). `va.1` (23 oct.) : +0.10 % vs v1 estimé honnêtement (optimisé sur 70 %
  des soirs, testé sur les 30 % jamais vus), +0.37 % sur ce qu'elle a vu. Le modèle de match n'est pas
  scanné : sur ~110 matchs, ça empirait de 3.4 % les matchs jamais vus.
- **Rétroactif pour tout le monde** : toutes les versions (auto comprises) sont rejouées sur toute la
  saison, sans réseau. Exception au JUGEMENT seulement : le Tuner et le Calibreur, ajustés sur les vieux
  soirs, comptent à partir de leurs soirs de validation (`juge_depuis`), sinon ils se noteraient sur
  leur propre corrigé. Garde-fous contre « apprendre le bruit par cœur » à force de choisir sur les
  mêmes soirs : une génération finit seulement après 3 soirs en direct, et une version auto devient
  principale seulement si elle tient aussi en direct (gain > 0 sur au moins 1000 joueurs).
- **Patterns** : segments où la principale se trompe systématiquement (écart >= 2.5 écarts-types).
- État : `data/versions_auto.json`. Pour retirer une version auto : y mettre `"retiree": true`.

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
