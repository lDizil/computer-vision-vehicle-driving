# Box refinement on fixed camera decisions

The reference detector, previous box refiner, and presence thresholds remain
unchanged. Only boxes already marked PRESENT may be adjusted in the offline
experiment. A crop does not add sensor information: it reuses the rectified
224x224 image and changes how many encoder tokens cover the local region.

## Exact inputs and roles

Use the existing appearance capture (its exact manifest SHA-256) and its
prior video/tracking captures and verified data packs. There are no new
archives or model weights in this experiment. The old tracking and
appearance evaluations have both been observed before. They are withheld
from this refiner's training and selection, not new blind holdouts:

| Role | Recorded packs | Frames |
|---|---|---:|
| Train | temporal development and stress | 288 |
| Dev | previous temporal holdout | 144 |
| Reference | earlier tracking and appearance holdouts | 144 + 144 |

Require disjoint recordings, day/vehicle groups, frame IDs, and JPEG/RGB
identities between all roles. Mount inputs read-only in the existing
alignment CPU image, with networking disabled. Keep artifacts outside every
input pack. Do not commit tensors, checkpoints, images, or target aliases.

1. **scripts.capture_spatial_zoom** takes the same pack/capture paths and digests
   as the appearance experiment and the existing visual-model root. It
   opens no label sidecars. Capture window-relative global and crop grids
   for the same nonempty source slots, permitting one fixed crop per
   PRESENT box. Save the printed byte hash.
2. **scripts.run_spatial_zoom** takes the same paths and digests plus the
   zoom capture and five truth files. Fit matched 0, full-frame-DINO and
   zoom-DINO heads on frozen features, with seeds 0 and 1, eight epochs,
   batches of 16. Geometry-only receives the same shape of zero features,
   same boxes/windows, loss, initialization and batch ordering.
3. Compare dev results and save the gate and all six checkpoints. Reload
   exact weights before opening reference labels. The benchmark can run
   again from captured grids without model weights. See --help for both
   commands.

## Gate and interpretation

Count one-to-one IoU >= .5 frame matches, false/missed assignments, matched
IoU and labeled area buckets (small: projected area below 1% of the image).
Both seeds must gain at least two matches over both the original and
same-seed geometry control and lose no more than one match on any dev
recording relative to baseline. Close-up must also gain two against global
DINO on both seeds. Prefer eligible close-up, then eligible global; use
seed 0 of the chosen family. Otherwise retain the original. Never choose
another model based on the reference metrics.

Rerun the unchanged zone and strict-motion tracking rules as secondary
checks on the adjusted geometry, not to select the refiner. The theoretical
fixed-match IDF1 ceiling uses twice the IoU match count divided by labeled
plus displayed boxes. It is not an official MOT result and not necessarily
achievable.

This experiment adds no driving control, 3D localization, or live tracking
changes. Require a separately prepared, unexposed recording set before
treating a dev/reference gain as ready to replace the stored refiner.
