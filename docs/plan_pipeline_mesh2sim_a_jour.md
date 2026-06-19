# Mesh2Sim, plan complet des pipelines mono et multicam (version à jour)

Document de référence vivant. Remplace pipeline_complet_etapes.md et plan_dev_pipeline_v2.md, mis à
jour de toutes les décisions prises pendant la construction (juin 2026). Pas de tiret cadratin.

Le projet : capture de mouvement sans marqueurs pour la biomécanique clinique, mono-caméra et
multi-caméra, un seul back-end (mesh MHR via SAM 3D Body) branché sur un pont vers OpenSim qui
produit des angles articulaires. Objectif : dépasser Pose2Sim, rester modulaire, valider sur 200
sujets (jeunes et âgés, marche et sit-to-stand) contre une référence optoélectronique.

Légende d'état : [FAIT] codé et testé, [DÉCIDÉ] tranché mais pas codé, [OUVERT] à creuser.

---

## Vue d'ensemble des étages (commun mono et multi)

Le pipeline se lit en deux temps : une phase de configuration hors traitement (faite une fois, ou
une fois par sujet), et la boucle de traitement (par frame). C'est une clarification importante
acquise pendant la construction.

Configuration (hors boucle de traitement) :
- C1. Carte de correspondance marqueurs vers vertices MHR. Une fois pour un modèle, via Mesh2Marker.
- C2. Génération du .osim personnalisé par sujet. Une fois par sujet, via Mesh2Marker, après que
  la forme définitive du sujet a été produite (par shape-lock en mono, par fusion puis shape-lock
  en multi).

Traitement (boucle) :
- T1. Frontend, inférence MHR par frame.
- T2. Cibles 2D conditionnelles (SynthPose).
- T3. Adapter, positions de marqueurs observées par frame.
- T4. Cohérence temporelle et shape-lock.
- T5. Fusion multi-vue (multi) ou assemblage (mono).
- T6. Deux composants distincts à ne pas confondre :
  - **T6a. Calibration des caméras** (multi seulement) : **prérequis amont** de T5, faite une fois
    à l'installation labo et réutilisée par sujet.
  - **T6b. Transformation de repère** vers monde Y-up OpenSim (mono et multi) : opération aval,
    après T3.
- T7. Raffinement de pose physique.
- T8. Pont biomeca : scaling, registration, IK.
- T9. Sortie : angles articulaires, et cinétique optionnelle.

Et transversal : V. Validation.

L'ordre exact diffère entre mono et multi. Voir la section suivante.

---

## Ordre des étages selon le mode

Les deux séquences ne sont **pas** identiques. Différences clés : la fusion multi-vue exige des
caméras calibrées en amont, et la forme définitive du sujet (qui alimente C2) sort du shape-lock
en mono mais de la fusion plus shape-lock en multi. À garder distinctes.

### Séquence MONO

Configuration préalable (faite une fois, hors session) :
- C1. Carte de correspondance C1 (un modèle par défaut fourni par Mesh2Marker, ou pickée une fois).

Boucle par sujet et par essai :
1. T1. Frontend MHR par frame, émet un BodyEstimate par frame.
2. T2. Cibles 2D conditionnelles, activées seulement où l'alignement 2D du mesh dégrade.
3. T4. Cohérence temporelle et shape-lock sur les frames mono. **Scales figés** (longueurs non
   observables). Produit la forme définitive du sujet.
4. **C2. Génération du .osim personnalisé** par sujet via Mesh2Marker, à partir de la forme
   verrouillée à l'étape précédente.
5. T3. Adapter frame par frame, AnatomicalObservation par frame (positions de marqueurs lues sur
   le mesh du sujet via mhr_vertices).
6. T5. Assemblage temporel des AnatomicalObservation en AnatomicalTrajectory (view_id unique).
7. T6b. Transformation de repère vers monde Y-up OpenSim (floor MoGe, lean, échelle par taille).
8. T7. Raffinement de pose physique.
9. T8. Pont biomeca : scaling, registration, IK sur le .osim personnalisé de l'étape 4.
10. T9. Sortie BiomechFit.

### Séquence MULTI

Configuration préalable (faite **une fois pour l'installation labo**, réutilisée pour tous les
sujets tant que les caméras n'ont pas bougé, ce n'est PAS un coût par sujet) :
- C1. Carte de correspondance.
- **T6a. Calibration des caméras** : extrinsèques markerless via lab-camera-dynamic-calibrator,
  synchronisation spatiotemporelle des vues. Intrinsèques connus ou calibrés ponctuellement.

Boucle par sujet et par essai :
1. T1. Frontend MHR par vue (inférence sur chaque caméra), un BodyEstimate par frame par vue.
2. T2. Cibles 2D recommandées (évidence indépendante qui rend la correction multi-vue réelle).
3. **T5. Fusion multi-vue type MAMMA** : fit MHR unique sur toutes les vues par énergie de
   reprojection 2D, **utilise les extrinsèques calibrées en T6a**. Pas de triangulation de
   marqueurs (topologie MHR fixe). Opération dans l'espace MHR, **pas besoin du .osim ici**.
   **Produit la meilleure estimation de forme et de pose du sujet** en exploitant toutes les vues.
4. T4. Shape-lock à partir de la forme fusionnée. **Scales libres faiblement régularisés** (la
   géométrie multi-vue observe les longueurs). Produit la forme définitive verrouillée.
5. **C2. Génération du .osim personnalisé** par sujet via Mesh2Marker, à partir de la forme
   fusionnée verrouillée.
6. T3. Adapter frame par frame, AnatomicalObservation par frame.
7. T6b. Transformation de repère vers monde Y-up OpenSim, applique les extrinsèques calibrées
   en T6a pour mapper le repère MHR fusionné vers le monde.
8. T7. Raffinement de pose physique.
9. T8. Pont biomeca : scaling, registration, IK sur le .osim personnalisé de l'étape 5.
10. T9. Sortie BiomechFit.

### Différences clés mono vs multi (rappel)

- **Calibration caméras** : prérequis amont en multi (une fois pour le labo), absente en mono.
- **Fusion T5** : MAMMA en multi (vient AVANT le shape-lock, produit la forme), simple
  assemblage temporel en mono (vient APRÈS le shape-lock, n'a pas de rôle de forme).
- **Source de la forme définitive qui alimente C2** : shape-lock T4 sur frames mono, vs
  shape-lock T4 alimenté par la fusion T5 en multi.
- **Scales pendant shape-lock** : figés en mono, libres faiblement régularisés en multi.

---

## Configuration

### C1. Carte de correspondance (Mesh2Marker, une fois par modèle) [FAIT côté outil]

Mesh2Marker (outil de config séparé, installable pip, repo distinct) charge le mesh MHR template et
le modèle .osim, permet de poser manuellement les marqueurs sur le mesh (picking de vertices,
centroïde possible), et exporte une CorrespondenceMap : pour chaque marqueur, les mhr_vertices
invariants, le segment OpenSim, l'offset local d'authoring, le flag bony/soft, l'index SynthPose
optionnel. Grâce à la topologie MHR fixe, cette carte est un actif réutilisable défini une seule
fois. Un modèle par défaut et sa carte seront fournis dans le repo Mesh2Marker.

État : l'outil est fonctionnel (coeur pur testé, couche Blender vérifiée dans le viewport). La
CorrespondenceMap exportée est conforme au contrat mesh2sim.contracts (validé field par field). Le
picking réel des 73 marqueurs reste à faire (travail humain). Les champs bony/soft et synthpose_index
ne sont pas encore peuplés par l'UI.

### C2. Génération du .osim personnalisé par sujet (Mesh2Marker, une fois par sujet) [DÉCIDÉ]

À partir de la **forme définitive verrouillée** du sujet (paramètres de forme MHR figés par le
shape-lock T4) et de la CorrespondenceMap, le pipeline appelle Mesh2Marker (fonction de
bibliothèque, coeur pur, generate.py) pour produire un .osim dont les marqueurs de référence
sont placés sur la peau du sujet, via les mêmes mhr_vertices appliqués au mesh du sujet.

Apport novateur : markerset de référence adapté à la morphologie individuelle, pas à un sujet
moyen. C'est un différenciateur vs OpenCap et BioPose (marqueurs fixes).

**Règle d'ordre unifiée** : C2 se fait une fois que la forme du sujet est définitive. L'étage
qui produit cette forme définitive diffère selon le mode :
- **Mono** : C2 vient après le shape-lock T4 sur les frames mono (scales figés, longueurs non
  observables).
- **Multi** : C2 vient après le shape-lock T4 alimenté par la fusion MAMMA T5 (scales libres
  faiblement régularisés, géométrie multi-vue observe les longueurs). La fusion produit
  l'estimation de forme la plus fiable en exploitant toutes les vues simultanément, donc
  générer le .osim avant elle gâcherait l'information multi-vue.

C2 ne précède donc jamais T1 (inférence), ni la fusion T5 en multi.

---

## Traitement

### T1. Frontend, inférence MHR [FAIT]

Entrée : vidéo (ou images) plus intrinsèques caméra optionnelles. Sortie : un BodyEstimate par frame
(mesh 18439 vertices, skeleton 127 joints, keypoints 2D, native_params MHR opaques), conforme au
contrat, sérialisable. Coeur d'inférence SAM 3D Body vendorisé (yangtiming, vanilla, pinné). Interface
MeshEstimator générique pour rester swappable. Multi-personnes via tracking, sélection du sujet
principal en clinique. Fingerprint de rig qui verrouille la cohérence d'énumération des joints.

Chaîne de scaling élucidée : l'inférence sort des params décomposés (shape 45, scale 28, body_pose
133, etc.), le rig reconstruit le vecteur 204 par transformation linéaire. Les 73 indices de scaling
isolés vivent dans les 68 scales PCA (67) plus 6 modes de forme en body_pose. Documenté dans
CLAUDE.md.

### T2. Cibles 2D conditionnelles, SynthPose [OUVERT]

Détecteur 2D anatomique indépendant (SynthPose, 52 keypoints). Rôle : fournir une évidence 2D
indépendante du mesh pour le raffinement et la fusion. DÉCISION : conditionnel en mono (seulement où
l'alignement 2D du mesh dégrade), recommandé en multi (évidence indépendante qui rend la correction
multi-vue réelle). Critère d'activation empirique, à fixer sur données. Le synthpose_index de la
carte fait le lien keypoint 2D vers marqueur. Pas prioritaire tant que le mono de base n'est pas
mesuré.

### T3. Adapter BodyEstimate vers AnatomicalObservation [PROCHAIN TICKET]

Le pipeline lit la CorrespondenceMap et, pour chaque frame, extrait la position 3D de chaque marqueur
sur le mesh du sujet (vertex unique ou centroïde des mhr_vertices). Produit un AnatomicalObservation
conforme (landmarks nommés, pos_3d, source bony/soft depuis le flag fixed, visibilité), dans le repère
du mesh MHR. C'est du traitement frame par frame, calcul simple (lecture de vertices, moyenne), fait
par le PIPELINE et pas par Mesh2Marker. Verrou : vérifier que le mhr_topology_id de la carte
correspond à la topologie du BodyEstimate, rejet bruyant sinon. Pas de transformation de repère ici
(en aval). Cohérence avec C2 garantie par le shape-lock (même morphologie partout).

### T4. Cohérence temporelle et shape-lock [DÉCIDÉ]

Verrouillage de la forme et des scalings du sujet sur l'essai (shape-lock), à partir des paramètres
de forme stabilisés. Lissage de la pose en espace latent MHR (Butterworth filtfilt). En mono, scales
figés (longueurs non observables). En multi, scales libres faiblement régularisés (géométrie observe
les longueurs). Le shape-lock est aussi ce qui aligne C2 (osim personnalisé) et T3 (observations).

### T5. Fusion multi-vue (multi) ou assemblage (mono) [DÉCIDÉ]

**Multi** : fit MHR unique sur toutes les vues par énergie de reprojection 2D, pondérée par
visibilité/incertitude par vue, façon MAMMA. Pas de triangulation de marqueurs, la topologie MHR
fixe donne la correspondance. Recouvre la profondeur et les rotations hors-plan que le mono ne
voit pas.

**Prérequis amont** : extrinsèques caméras calibrées et vues synchronisées (T6a, faite une fois
à l'installation labo et réutilisée par sujet). Sans cette calibration, T5 ne peut pas reprojeter
le mesh dans chaque vue, donc ne peut pas fonctionner.

**Ordre dans la séquence multi** : T5 vient AVANT T4 (shape-lock). La fusion produit la
meilleure estimation de forme du sujet à partir de toutes les vues, puis le shape-lock la
verrouille, puis C2 génère le .osim. Opération dans l'espace MHR : pas besoin du .osim ici,
il sera généré après.

**Mono** : simple assemblage temporel des AnatomicalObservation en une AnatomicalTrajectory,
view_id unique. Vient APRÈS T4 (shape-lock) et T3 (adapter), aucun rôle de forme.

Sortie commune : AnatomicalTrajectory (positions T x L x 3 en repère monde, shape_descriptor
unique pour l'essai).

### T6. Calibration des caméras (amont, multi) et transformation de repère (aval, mono + multi) [DÉCIDÉ]

À distinguer en deux composants. Ils étaient confondus dans la première version du plan, ce qui
amenait à les placer au même endroit. C'est faux : la calibration est un prérequis amont en
multi, la transformation de repère est aval dans les deux modes.

**T6a. Calibration des caméras (multi seulement, PRÉREQUIS AMONT de T5)** :
- Extrinsèques markerless via lab-camera-dynamic-calibrator (composant éprouvé dans
  FastSAM3DToOpenSim, à réécrire dans cet étage et pas dans le frontend).
- Synchronisation spatiotemporelle des vues.
- Intrinsèques connus ou calibrés ponctuellement.
- **Faite une fois à l'installation labo**, réutilisée pour tous les sujets et tous les essais
  tant que les caméras n'ont pas bougé. Pas un coût par sujet.
- Sans elle, la fusion T5 par reprojection ne peut pas fonctionner.
- En mono : non applicable (pas de fusion par reprojection).

**T6b. Transformation de repère vers monde Y-up OpenSim (mono et multi, AVAL de T3)** :
- Mono : floor via MoGe, correction de lean, échelle par taille.
- Multi : applique les extrinsèques calibrées en T6a pour mapper le repère MHR fusionné vers
  le repère monde OpenSim.
- Reprend des composants éprouvés de FastSAM3DToOpenSim (CoordinateTransformer).

### T7. Raffinement de pose physique [DÉCIDÉ, nouveau]

Étage inspiré d'OpenCap Monocular, identifié comme crucial. Entre la sortie mesh et l'IK, optimisation
qui minimise l'erreur de reprojection 2D, le glissement et la pénétration des pieds au sol, la vitesse
articulaire excessive. Corrige la dérive du bassin (eux : 56.9cm vers 4.9cm sur 5 STS) et améliore
fortement la précision (eux : -48% rotation, -69% translation vs CV brut plus IK). Poids de départ par
activité repris de leur Table S1 (filtre 6Hz marche / 4Hz STS, poids reprojection 250 en STS, etc.).
Lié à la détection d'activité (voir transverse).

### T8. Pont biomeca, scaling plus registration plus IK [DÉCIDÉ, gros morceau]

Le coeur scientifique. Trois sous-étages selon l'arbitrage moteur :
- Scaling : mise à l'échelle du modèle. Mono scales figés, multi libres régularisés. S'appuie sur la
  chaîne de scaling MHR élucidée (28 scale_params vers 68 scales PCA).
- Registration des offsets de marqueurs : correspondance MHR vers anatomie en deux étages. Carte
  générique calibrée une fois sur dataset annoté disjoint puis figée/fortement régularisée, plus
  petit résidu par sujet contraint. Verrou central du projet (identifiabilité). Marqueurs bony figés,
  soft plus libres (flag fixed).
- IK trajectoire : résolution des angles sur le modèle personnalisé avec les trajectoires observées.
Arbitrage moteur : stage A (fit MHR) PyTorch ; B1 (scaling plus registration, une fois par sujet)
nimblephysics ; B2 (IK à l'échelle) nimblephysics, repli MJX/MuJoCo si goulot GPU. Code de départ du
pont : SMPL2AddBiomechanics (Keller), entrée SMPL remplacée par points MHR.

### T9. Sortie [DÉCIDÉ]

BiomechFit : modèle scalé (.osim), angles par DoF (.mot), offsets recalés, résidus, incertitude
optionnelle par DoF (style Cotton). Cinétique optionnelle en aval (GRF, moments) pour usages
cliniques (sit-to-stand, KAM), pas le coeur de la thèse mais bonus.

---

## Transverse

### Détection d'activité [DÉCIDÉ, nouveau]

Classifier l'activité (marche, squat, sit-to-stand) pour adapter les paramètres de traitement (poids
d'optimisation T7, fréquence de filtre, niveau de détail MHR). Inspiré d'OpenCap (Video-LLaMA3).
ATTENTION : adapter les paramètres de traitement, mais garder le modèle OpenSim cible constant pour ne
pas casser la validation à modèle unique. Distinguer adapter le traitement (ok) de changer le modèle
de sortie (à éviter).

### V. Validation [DÉCIDÉ, colonne vertébrale]

200 sujets, jeunes et âgés, marche et sit-to-stand, gold standard optoélectronique synchronisé.
Règles : markerless et référence dans le MÊME modèle OpenSim et même IK. Opto pas vérité terrain
(artefact tissu mou, pire sur rotations) donc Bland-Altman, pas erreur contre vérité. Cibles
stratifiées par DoF et tâche, pas un 5 degrés uniforme. ICC(A,1) accord absolu. n dimensionné sur
largeur d'IC des limites d'agreement. Trois bras de comparaison : keypoints (COCO) vers IK, keypoints
anatomiques (SynthPose) vers IK, et mesh (le pipeline), pour défendre l'apport du mesh sur les
rotations hors-plan. Baselines externes à battre : OpenCap Monocular (4.8 deg rotation, mono SMPL,
jeunes seulement), BioPose (NeurIK appris).

---

## Différences mono vs multi (tableau)

| Aspect | Mono | Multi |
| --- | --- | --- |
| Vues | 1 (view_id mono) | N caméras calibrées synchronisées |
| Fusion (T5) | assemblage temporel | fit MHR unique multi-vue par reprojection |
| Scales (T4) | figés (longueurs non observables) | libres faiblement régularisés |
| Profondeur / rotations hors-plan | faibles, dépend du prior | recouvrées par intersection de rayons |
| Calibration (T6) | floor MoGe, lean, échelle par taille | extrinsèque markerless plus synchro |
| Cibles 2D (T2) | conditionnel | recommandé |
| Usage | déployable, Synkro prod | précision, référence |

---

## État d'avancement global

Fait : socle monorepo, contrats plus vocabulaire réel plus CorrespondenceMap, probe MHR et chaîne de
scaling, frontend T1 de bout en bout. Outil Mesh2Marker fonctionnel et conforme au contrat (C1).

Prochain ticket : adapter T3.

Décidé pas codé : C2, T4, T5, T6, T7, T8, T9, détection d'activité, validation.

Ouvert : critère d'activation SynthPose (T2), représentation temporelle fine (T4), protocole de
synchronisation optoélectronique (irréversible, à figer avant collecte), plan d'analyse statistique
détaillé, test du biais de forme MHR vs anthropométrie réelle sur les âgés.

Prérequis dormants à traiter tôt : localisation dépôt et licence Cotton, design de collecte de
données (irréversible).

Prérequis résolus :
- Test d'installation nimblephysics sur stack 2026 [RÉSOLU 2026-06-18]. Utilisable sur Python 3.11
  (wheel manylinux_2_28 cp311 de la release 0.10.52.1). Le sous-module `nimble.biomechanics` expose
  toutes les classes critiques pour T8 (`MarkerFitter`, `OpenSimParser`, `IKErrorReport`,
  `DynamicsFitter`, `SkeletonConverter`). L'arbitrage moteur tient : PyTorch pour A, nimblephysics
  pour B1/B2. MJX/MuJoCo reste un repli optionnel **non bloquant**. Note pour la construction de
  l'étage `biomech_bridge` : il aura son propre environnement isolé (différent de frontend) ; à ce
  moment-là, évaluer une variante torch CPU-only pour alléger la traîne nvidia tirée par défaut
  (~700 MB) si nimble n'exige pas l'accélération GPU côté solveur.

---

## Garde-fous transverses

- La validation 200 sujets est la contribution. Tout est à son service.
- La fidélité benchmark d'un modèle n'est pas la précision angulaire clinique. Le harnais maison
  arbitre, pas arXiv.
- Front-end swappable : méthode et validation survivent au remplacement du modèle de mesh.
- Mesh2Marker est un outil de configuration, pas un moteur de traitement. Lié par le fichier de
  correspondance et un appel de bibliothèque ponctuel (C2), jamais dans la boucle frame par frame.
- Un environnement par étage lourd, contrats légers partagés. Cloisonnement strict.
