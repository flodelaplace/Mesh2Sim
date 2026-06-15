# Mesh2Sim, contexte projet pour Claude Code

Pipeline de capture de mouvement sans marqueurs pour la biomécanique clinique, en version
mono-caméra et multi-caméra, avec un seul back-end (estimation d'un mesh MHR via SAM 3D Body) qui se
branche sur un pont vers OpenSim produisant des angles articulaires. But final : angles précis,
validés sur 200 sujets (jeunes et âgés, marche et sit-to-stand) contre une référence
optoélectronique. Dépasser Pose2Sim, rester modulaire face au renouvellement des modèles.

Tagline : markerless mesh-based biomechanics, monocular and multi-view, from video to OpenSim.

## Décisions verrouillées

- Front-end : Fast-SAM-3D-Body vers MHR, conçu comme composant interchangeable.
- Fusion multi-vue type MAMMA, sans réentraîner de détecteur dense (topologie MHR fixe donne la
  correspondance gratuitement). Pas de triangulation de marqueurs.
- Mise à l'échelle asymétrique : figée en mono, libre faiblement régularisée en multi.
- Pont biomeca par recalage des marqueurs sur le modèle OpenSim (échelle plus offsets plus IK).
- Correspondance MHR vers anatomie en deux étages : carte générique calibrée une fois sur dataset
  disjoint puis figée, plus petit résidu par sujet. Verrou central du projet.
- Arbitrage moteur : PyTorch pour le fit du mesh, nimblephysics pour le recalage, MuJoCo/MJX en
  repli GPU.
- Règle de validation à modèle unique : markerless et référence dans le même modèle OpenSim.
- Épine dorsale modulaire : un contrat de données anatomique (points nommés sur vocabulaire MoCap),
  indépendant du modèle de mesh. Tout le reste se branche dessus. Voir docs/contracts_spec.md.

## Garde-fous d'ingénierie (cruciaux)

- Les dépendances sont mutuellement incompatibles. NE JAMAIS installer dans le même environnement :
  nimblephysics (Python 3.9, figé fin 2024) et le stack 3DB (PyTorch CUDA récent), ni JAX (MJX) et
  PyTorch CUDA, ni l'écosystème mmcv (SynthPose) avec le reste. Un environnement par étage.
- Le seul composant partagé est le package contracts, gardé ultra-léger (pydantic plus numpy),
  installable proprement dans tous les environnements.
- Les étages communiquent par fichiers (contrats sérialisés), pas par imports croisés. Les étages ne
  s'importent jamais entre eux. L'orchestrateur traverse les environnements par sous-processus ou
  conteneurs.
- Plateforme : développement dans WSL2, exécution des étages en conteneurs Docker construits depuis
  WSL2. GPU via le NVIDIA Container Toolkit.
- Ne pas réécrire le projet existant FastSAM3DToOpenSim. C'est un repo à part, baseline et réservoir
  de composants à récupérer (CoordinateTransformer avec MoGe et lean, BoT-SORT, lecture des joints
  MHR).

## Conventions

- Unités : mètres, pixels, radians en interne (degrés à l'affichage).
- Repère monde : Y vers le haut, main droite, cohérent avec OpenSim.
- Rotations en matrices 3x3 dans les contrats.
- schema_version dans chaque contrat.

## Garde-fous de raisonnement

- La fidélité benchmark d'un modèle n'est pas la précision angulaire clinique. Le harnais de
  validation maison est l'arbitre.
- L'optoélectronique n'est pas une vérité terrain (artefact de tissu mou). Cadrer en accord de
  méthodes (Bland-Altman).
- Cibles de précision différenciées par DoF et par tâche, jamais un 5 degrés uniforme.
- Le pont mesh vers OpenSim appris existe déjà (BioPose, arXiv 2501.07800, sur SMPL, mono). La
  différenciation de ce projet : MHR, multi-vue, recalage par sujet, validation 200 sujets. Inclure
  une baseline de type pont appris dans la validation.

## Façon de travailler

- Un environnement par étage, écrit (environment.yml plus Dockerfile) avant le code, versions
  épinglées.
- Tests d'abord aux frontières de contrat, golden tests pour la régression de bout en bout.
- Commits petits et atomiques, une préoccupation chacun.
