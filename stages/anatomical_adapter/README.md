# stages/anatomical_adapter

Étage léger qui héberge **deux étapes du pont biomeca en mono** :

- **T3** — adapter `BodyEstimate` (mesh MHR du sujet, par frame) + `CorrespondenceMap`
  → `AnatomicalObservation` (landmarks anatomiques nommés, par frame).
- **T5 mono** — assembler une séquence d'`AnatomicalObservation` → une
  `AnatomicalTrajectory` unique pour l'essai.

Les deux sont pure-numpy + `mesh2sim.contracts`, sans GPU, sans rig.

## Périmètre strict (T3 et T5)

T3 :
- ✅ Lookup vertex MHR via `marker.mhr_vertices`, centroïde si plusieurs.
- ❌ **Pas** de transformation de repère / scaling / IK / fusion.
- ❌ **Pas** d'utilisation de `marker.local_offset` ni `correspondence_map.frame_alignment`.

T5 mono :
- ✅ Tri temporel (timestamp prioritaire, fallback frame_id), monotonie stricte.
- ✅ Vérification d'ensemble de landmarks identique entre frames.
- ✅ Empilage `(T, L, 3)` + masque de visibilité `(T, L)`.
- ✅ **Pas** d'interpolation ni de lissage — c'est T7. Manquants → NaN/0.
- ✅ **Pas** de transformation de repère — c'est T6b. Positions stockées
   dans le repère d'origine des observations (typiquement `Pos3DFrame.camera`).
- ✅ Attache le `ShapeDescriptor` unique de l'essai (issu de T4).

## Verrous de sécurité

T3 :
- `TopologyMismatchError` si `correspondence_map.mhr_topology_id ≠ body_estimate.mesh.topology_id`.
- `VertexOutOfBoundsError` si un `mhr_vertex` sort de `[0, V)`.

T5 mono :
- `TemporalOrderError` sur séquence vide, timestamps non monotones ou doublons.
- `LandmarkSetMismatchError` si l'ensemble de landmarks diffère entre frames.
- `ViewMismatchError` si plusieurs `view_id` apparaissent en mono.

## Environnement

Léger : Python 3.11 + numpy + `mesh2sim-contracts`. Pas de torch, pas de CUDA, pas de
nimblephysics. Construire :

```bash
conda env create -f environment.yml
conda activate mesh2sim-anatomical-adapter
```

## API publique

```python
from mesh2sim.contracts import BodyEstimate, CorrespondenceMap, ShapeDescriptor, load, save
from mesh2sim_anatomical_adapter import (
    body_estimate_to_anatomical_observation,   # T3
    assemble_trajectory_mono,                   # T5 mono
)

# T3: per-frame adapter
be = load(BodyEstimate, "path/to/body_estimate_dir")
cm = load(CorrespondenceMap, "path/to/correspondence_map_dir")
obs = body_estimate_to_anatomical_observation(be, cm)

# T5 mono: per-trial assembly
shape_descriptor: ShapeDescriptor = ...  # from T4 shape-lock
traj = assemble_trajectory_mono(
    [obs0, obs1, obs2, ...],
    shape_descriptor,
    subject_id="S001", trial_id="walk_01", task="gait", fps=30.0,
)
save(traj, "path/to/output_dir")
```

## Repère

Les positions extraites restent dans le **repère du mesh MHR** (proche du repère
caméra, post-inférence mais avant T6b). T5 ne transforme rien ; `AnatomicalTrajectory`
n'a pas de champ explicite pour le repère, donc T5 stash `assembled_from_pos3d_frame`
dans `provenance.extra` pour traçabilité. La transformation vers le monde Y-up OpenSim
est faite par T6b en aval.

## Tests

```bash
pytest tests -v
```

Couvre : les 3 cas de calcul T3, les verrous T3, le round-trip T3, l'assemblage nominal
T5, le réordonnancement temporel, la préservation des NaN aux marqueurs manquants, les
4 verrous T5, l'attachement du ShapeDescriptor, et les round-trips T5 (incluant la
préservation des NaN à travers l'IO).
