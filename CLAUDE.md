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

## Paramètres MHR et chaîne de scaling

Acquis vérifié par investigation (probe + lecture du coeur vendorisé + reconstruction
empirique). Tous les indices et tailles ci-dessous sont confirmés par exécution réelle.

- SAM 3D Body sort des paramètres **décomposés**, pas le vecteur consolidé de 204. Sorties
  par personne dans `process_one_image` : `shape_params` (45), `scale_params` (28),
  `body_pose_params` (133), `hand_pose_params` (108), `global_rot` (3), `expr_params` (72),
  `pred_pose_raw` (266), plus `pred_cam_t` (3) qui sert de proxy pour `global_trans`.
- Le rig MHR consomme un vecteur `model_parameters` de dimension 204, reconstruit par une
  transformation linéaire dans `mhr_head._mhr_forward_core` :
  - `[0, 3)`   : `global_trans × 10` (mise à l'échelle pour stabilité d'optimisation en mètres)
  - `[3, 6)`   : `global_rot`
  - `[6, 136)` : `body_pose_params[0:130]` (le réseau sort 133, les 3 derniers droppés ; les
    positions hand sont en option écrasées par `hand_pose_params` décodé si
    `enable_hand_model=True`, désactivé dans notre env)
  - `[136, 204)` : 68 `scales` PCA-décodés, par `scales = scale_mean + scale_params @ scale_comps`
    où `scale_mean` a forme `(68,)` et `scale_comps` a forme `(28, 68)`
- `shape_params` (45) et `expr_params` (72) **ne sont pas** dans le 204. Ils sont passés
  séparément au forward du rig comme `identity_coeffs` et `face_expr_coeffs`. Ils ne sont
  donc pas sujets au probe de scaling qui ne perturbe que les 204.
- Les 73 indices de scaling isolés par le probe (`scaling_mask_v2.json`) se répartissent
  en deux composants distincts :
  - 67 dans le vecteur `scales` PCA-décodé (longueurs de segment morphologiques, espace
    dimension 68, à l'exception de la position vestigiale `scales[15]` = indice 204[151]
    à exclure car son effet est sous le bruit float32)
  - 6 dans `body_pose_params` aux positions 124..129 (modes de forme PCA encodés comme
    translations 1-DoF dans l'espace body_pose ; indices 204 = 130..135). Structurellement
    ce sont des modes coordonnés multi-jointures via `parameter_transform`, pas des
    paramètres de scaling au sens segment-par-segment.
- Les 4 `pose_dof` exclus du masque de scaling sont : indices 0, 1, 2 (translation racine,
  alimentée par `pred_cam_t`) et 62 (rotation simple d'articulation du squelette).
- Conséquence pour le pont OpenSim : piloter les scalings depuis l'inférence réelle part
  des 28 `scale_params` et applique la PCA pour obtenir les 68 `scales`. Les 6 modes de
  forme passent par la matrice `parameter_transform`, pas directement. La correspondance
  vers le scaling OpenSim segment-par-segment **n'est pas un-pour-un** : ce sera une
  optimisation, traitée comme une registration biomeca par sujet.
- Table complète indice-par-indice : `stages/frontend_mhr/results/204_to_components.json`.
  Probe + désambiguation : `stages/frontend_mhr/results/{scaling_mask_v2.json,disambiguation.json}`.

## Rôle de Mesh2Marker et ordre du bridge

Mesh2Marker est un **outil de configuration** installable par pip, **pas un moteur de
traitement**. Il ne traite jamais de vidéo ni de frame par frame. Il intervient à deux
moments précis, hors de la boucle de traitement :

1. **Configuration globale, une fois par modèle OpenSim**. Pose manuelle des marqueurs sur
   le mesh MHR, produit la `CorrespondenceMap` (marqueurs → `mhr_vertices` invariants en
   topologie). Le repo Mesh2Marker fournit un modèle par défaut et sa carte pour éviter de
   refaire l'étape.
2. **Personnalisation par sujet, une fois par participant, APRÈS l'inférence**. À partir
   des paramètres morphologiques du mesh du sujet (issus de l'inférence : `shape_params`,
   `scale_params`) et de la `CorrespondenceMap`, Mesh2Marker génère un `.osim` personnalisé
   avec les marqueurs de référence placés sur la peau du sujet. Cette étape **ne peut pas
   précéder l'inférence** puisqu'elle consomme les paramètres de forme produits par
   l'inférence. Apport novateur : markerset de référence adapté à la morphologie
   individuelle, pas à un sujet moyen.

Le **traitement frame par frame** (extraction des positions de marqueurs observées sur
les meshes du sujet via les `mhr_vertices` de la carte, centroïde si plusieurs vertices
par marqueur) est fait par le **pipeline**, pas par Mesh2Marker. C'est un calcul simple
(lecture de vertices, moyenne) qui appartient à l'adapter
`BodyEstimate → AnatomicalObservation`.

**Cohérence garantie par le shape-lock** : la forme du sujet est verrouillée sur l'essai,
donc le `.osim` personnalisé (étape 2 ci-dessus) et les meshes traités frame par frame
partent de la même morphologie. Modèle de référence et observations sont alignés par
construction, sans correction post-hoc.

**Ordre des étages du bridge**, à respecter strictement :

1. Front-end MHR : inférence par frame, émet un `BodyEstimate` par frame (et par vue).
2. Shape-lock : extraction et figeage des paramètres morphologiques pour l'essai.
3. Génération du `.osim` personnalisé par sujet via Mesh2Marker (étape 2 du paragraphe
   précédent), à partir du shape verrouillé et de la `CorrespondenceMap`.
4. Adapter frame par frame : `BodyEstimate → AnatomicalObservation` (positions observées
   des marqueurs en lisant les `mhr_vertices` sur le mesh de chaque frame, centroïde si
   plusieurs).
5. Fusion multi-vue et assemblage temporel, transformation de repère vers le monde Y-up.
6. Scale + IK OpenSim sur le `.osim` personnalisé issu de l'étape 3.

## Dépendances à résoudre avant C2

Note durable, acquis le 2026-06-18 en explorant le repo Mesh2Marker (`/home/fdela/Mesh2Marker`)
pour préparer T4. À ne pas reperdre, parce que sans résolution la valeur ajoutée de C2
(`.osim` personnalisé adapté à la morphologie individuelle) s'effondre.

**Le problème** : Mesh2Marker exporte une base de forme MHR (`local_models/mhr_shape_basis.npz`)
qui ne contient que les 45 composantes d'identité (`dV (45, 18439, 3)`, `dJ`, `dKP`). Sa
fonction `morph(basis, betas)` (dans `core/src/mesh2marker/morph.py`) régénère le rest pose à
partir de 45 betas — c'est-à-dire de `shape_params` uniquement.

Or, d'après notre investigation `disambiguate_suspects.py` documentée plus haut, **67 des 73
scalings d'un sujet vivent dans les 68 scales PCA-décodés** (issus de `scale_params (28) @
scale_comps (28, 68) + scale_mean (68,)`), pas dans les 45 identités. En l'état, le `.osim`
personnalisé généré par C2 capturerait au mieux ~6 scalings sur 73 (les modes de forme en
body_pose) et raterait l'essentiel de la morphologie individuelle du sujet.

**Deux pistes à arbitrer côté Mesh2Marker, avant de construire C2** :

- **(a)** Étendre `mhr_shape_basis.npz` pour inclure une seconde base liée au pathway scale.
  Soit une base sur les 28 `scale_params`, soit une base sur les 68 `scales` post-PCA. La
  fonction `morph` doit alors accepter une seconde séquence de coefficients (`scale_betas`)
  et appliquer les deux contributions linéaires au rest pose. Avantage : pipeline C2 reste
  pur numpy (zero-wheel), Mesh2Marker garde son rôle.

- **(b)** Garder la base actuelle limitée à l'identité (le mesh template reste rest-pose
  default-scale) et appliquer les scales **directement sur le `.osim`** une fois généré, en
  redimensionnant les segments OpenSim avec les facteurs d'échelle reconstruits depuis les
  scale_params verrouillés. Avantage : pas de modification de Mesh2Marker. Inconvénient : la
  cohérence entre la peau du sujet (mesh) et le modèle (osim) doit être garantie côté pipeline,
  pas par Mesh2Marker.

T4 n'est pas concerné par ce flag : T4 régénère les meshes via le forward MHR complet du rig
vendoré (qui gère scale_params), donc les meshes locked-shape sont corrects pour T3 et la
suite. Le flag concerne uniquement C2 (génération du `.osim` personnalisé), qui passe par
Mesh2Marker.

## Façon de travailler

- Un environnement par étage, écrit (environment.yml plus Dockerfile) avant le code, versions
  épinglées.
- Tests d'abord aux frontières de contrat, golden tests pour la régression de bout en bout.
- Commits petits et atomiques, une préoccupation chacun.
