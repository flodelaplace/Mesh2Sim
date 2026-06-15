# Mesh2Sim, spécification du package contracts (l'épine dorsale)

Spec d'implémentation des schémas de données qui circulent entre les étages. Tout étage dépend de
ce package et de rien d'autre des autres étages. À placer dans docs/ du repo et à donner à Claude
Code pour le Ticket 1.

Package : mesh2sim.contracts. Dépendances autorisées : pydantic (v2), numpy. RIEN d'autre de lourd.

Pas de tiret cadratin.

---

## Principes de conception

1. Sémantique anatomique, pas spécifique au modèle. Les contrats nomment des points anatomiques sur
   un vocabulaire commun, pas des indices de vertices MHR. Un nouveau modèle de mesh se branche via
   un adaptateur, sans toucher aux contrats.
2. Coeur requis plus capacités optionnelles. Chaque contrat a un noyau obligatoire et des champs
   optionnels signalés par des drapeaux de capacité. Les étages aval utilisent ce qui est présent et
   se dégradent proprement.
3. Tableaux en binaire, métadonnées en JSON. Les gros tableaux (mesh, trajectoires) en npz, les
   métadonnées et la structure en JSON manifeste. Jamais de gros tableaux en JSON.
4. Mono = multi avec N=1. view_id existe toujours, même en monoculaire, pour unifier les chemins de
   code.
5. Conventions fixées une fois pour toutes (voir plus bas), parce qu'une erreur de repère casse tout
   en silence.

## Conventions globales (à figer maintenant)

- Unités : mètres pour les positions 3D, pixels pour la 2D, radians pour les angles internes,
  degrés seulement à l'affichage et au rapport.
- Repère monde : Y vers le haut, main droite, cohérent avec OpenSim et avec le CoordinateTransformer
  existant de FastSAM3DToOpenSim.
- Ordre des tableaux : (T, ...) avec T le temps en premier pour les trajectoires.
- Rotations stockées en matrices 3x3 (pas d'ambiguïté d'ordre d'Euler dans les contrats ; les angles
  d'Euler n'apparaissent qu'en sortie biomeca avec convention explicite par DoF).
- Tout contrat porte un champ schema_version.

---

## Le vocabulaire (mesh2sim/contracts/vocab.py)

- LANDMARKS : ensemble figé de noms de marqueurs anatomiques, basé sur le set MoCap standard
  (référence : set OpenCap 43 marqueurs). Exemples : r_asis, l_asis, r_knee_med, r_knee_lat,
  r_ankle_med, r_ankle_lat, etc. C'est le vocabulaire partagé entre SynthPose, vertices MHR et
  marqueurs OpenSim.
- SEGMENTS : noms des segments OpenSim cibles (pelvis, femur_r, tibia_r, ...), par modèle supporté
  (Rajagopal2016, LaiUhlrich, modèle SportFX). Une table par modèle.
- Validation : tout nom de landmark utilisé dans un contrat doit appartenir à LANDMARKS.

---

## Schéma 1, BodyEstimate (sortie native d'un estimateur, par frame, par vue)

Représente ce qu'un modèle de mesh produit, avant standardisation. Spécifique à l'estimateur.

Champs :
- schema_version: str
- estimator_id: str (ex. "fast-sam-3d-body@0.x")
- frame_id: int
- view_id: str (id de caméra ; "mono" si vue unique)
- timestamp: float | None
- capabilities: Capabilities (has_mesh, has_skeleton, has_2d_keypoints, has_native_params)
- native_params: dict | None (opaque, spécifique modèle ; pour MHR : identity 45, model_parameters
  204, expression 72 ; jamais lu par les étages aval, seulement par l'adaptateur)
- mesh: MeshData | None (vertices (V,3) float32, topology_id: str, faces optionnel)
- skeleton_state: SkeletonState | None (joint_positions (J,3), joint_orientations (J,3,3),
  joint_names: list[str])
- keypoints_2d: Keypoints2D | None (names: list[str], xy (K,2), confidence (K,))
- camera: CameraParams | None
- frame_shape: tuple[int,int] | None (H, W)

## Schéma 2, CameraParams

- view_id: str
- K: (3,3) float (intrinsèques)
- distortion: (n,) float | None
- R: (3,3) float | None (extrinsèques, repère monde, rempli après calibration)
- t: (3,) float | None
- resolution: tuple[int,int] (W, H)
- time_offset: float | None (offset de synchro relatif, en secondes)

## Schéma 3, AnatomicalObservation (LE contrat clé, par frame, par vue)

Sortie standardisée d'un adaptateur (BodyEstimate plus carte de correspondance vers ce contrat).
C'est ce que consomment fusion et pont biomeca.

Champs :
- schema_version: str
- frame_id: int
- view_id: str
- timestamp: float | None
- landmarks: dict[str, Landmark] (clé dans LANDMARKS)
- pos3d_frame: enum {none, camera, world} (repère des pos_3d des landmarks)
- segment_frames: dict[str, (3,3)] | None (orientation de segment, capacité optionnelle)
- shape_descriptor: ShapeDescriptor | None (init d'échelle des segments OpenSim, pour bootstrap du
  scaling ; ex. vecteur d'échelles par segment dérivé du sous-vecteur scalings MHR)
- joint_centers_init: dict[str, (3,)] | None (init des centres articulaires pour l'IK, depuis
  skeleton_state)
- dense_surface: MeshData | None (mesh si un étage aval veut la densité)
- capabilities: Capabilities
- provenance: Provenance (estimator_id, correspondence_map_id, adapter_id)

Landmark (sous-objet) :
- pos_3d: (3,) float | None
- pos_2d: (2,) float | None
- confidence: float
- visibility: float (0 à 1)
- source: enum {bony, soft, unknown}

## Schéma 4, AnatomicalTrajectory (par essai, fusionné ou assemblé)

Sortie de la fusion multi-vue (multi) ou de l'assemblage temporel (mono). En repère monde.

Champs :
- schema_version: str
- subject_id: str
- trial_id: str
- task: enum {gait, sts, other}
- mode: enum {mono, multi}
- fps: float
- landmark_names: list[str] (sous-ensemble de LANDMARKS, ordre fixe)
- positions: (T, L, 3) float (repère monde, Y-up, mètres)
- confidence: (T, L) float
- shape_descriptor: ShapeDescriptor (UNIQUE pour l'essai, c'est le shape-lock : échelle verrouillée)
- views_used: list[str] (multi) ou ["mono"]
- uncertainty: (T, L, 3) float | None
- provenance: Provenance

## Schéma 5, BiomechFit (par sujet/essai, sortie du pont)

Champs :
- schema_version: str
- subject_id: str
- trial_id: str
- model_id: str (Rajagopal2016, LaiUhlrich, sportfx)
- scaled_model_path: str (.osim scalé)
- dof_names: list[str]
- angles: (T, D) float (radians) OU motion_path: str (.mot) ; au moins l'un des deux
- marker_offsets: dict[str, (3,)] (offsets recalés, registration)
- marker_residuals: (T,) float (erreur de fit marqueurs par frame)
- uncertainty: (T, D, 2) float | None (intervalle de confiance par DoF, style Cotton)
- provenance: Provenance

## Objets transverses

- Capabilities: bool flags (has_mesh, has_skeleton, has_2d_keypoints, has_native_params,
  has_segment_frames, has_shape_descriptor)
- Provenance: estimator_id, adapter_id, correspondence_map_id, created_at, schema_version, plus un
  champ libre extra: dict
- ShapeDescriptor: representation: enum {per_segment_scale, opaque}, data (dict segment->scale (3,)
  ou vecteur), source_model: str
- MeshData: vertices (V,3) float32, topology_id: str, faces (F,3) int | None
- SkeletonState, Keypoints2D : voir Schéma 1

---

## IO (mesh2sim/contracts/io.py)

- Chaque contrat sait se sérialiser et se désérialiser. Format : un fichier .npz pour les tableaux
  plus un manifeste JSON pour la structure et les métadonnées, ou un conteneur unique documenté.
- Fonctions : save(contract, path), load(cls, path). Round-trip exact garanti (un test par contrat).
- Validation au load : vérifier schema_version compatible, noms de landmarks dans LANDMARKS,
  cohérence des drapeaux de capacité avec les champs présents.
- Pour les séquences (AnatomicalObservation par frame sur un essai), prévoir un format batch
  efficace (parquet pour les landmarks tabulaires, npz pour les meshes), pour ne pas écrire des
  milliers de petits fichiers.

## Versionnement (mesh2sim/contracts/version.py)

- SCHEMA_VERSION: str, semver. Embarqué dans chaque contrat sérialisé.
- Règle de compatibilité au load (majeure identique requise, mineure tolérée).

## Tests (contracts/tests/)

- Round-trip save puis load identique pour chaque schéma.
- Rejet d'un nom de landmark hors vocabulaire.
- Cohérence capacité vs champ présent (ex. has_mesh True implique mesh non None).
- Validation des conventions (formes de tableaux, repère).

---

## Squelette d'ancrage pour AnatomicalObservation (pydantic v2, indicatif)

```python
from enum import Enum
import numpy as np
from pydantic import BaseModel, ConfigDict

class Source(str, Enum):
    bony = "bony"; soft = "soft"; unknown = "unknown"

class Landmark(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pos_3d: np.ndarray | None = None   # (3,)
    pos_2d: np.ndarray | None = None   # (2,)
    confidence: float = 0.0
    visibility: float = 0.0
    source: Source = Source.unknown

class AnatomicalObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    schema_version: str
    frame_id: int
    view_id: str
    timestamp: float | None = None
    landmarks: dict[str, Landmark]
    pos3d_frame: str = "none"          # none | camera | world
    # ... capacités optionnelles, provenance
```

Note pour Claude Code : pydantic v2 ne sérialise pas les np.ndarray nativement, donc l'IO doit gérer
les tableaux séparément (npz) et ne garder dans le JSON que les métadonnées et les chemins. Ne pas
mettre de gros tableaux dans le modèle sérialisé en JSON.
