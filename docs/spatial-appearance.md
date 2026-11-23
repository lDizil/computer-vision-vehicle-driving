# Appearance on top of frozen camera association

This is an attribution experiment, not a detector or a control input. The
existing frozen camera model and presence policy remain the same. All arms
use the exact original boxes, raw scores, and present/absent/uncertain states.
Only track numbers can change.

## Data and captures

The four already observed video packs (576 frames) serve as development data.
The new appearance_holdout pack on OpenScene camera shard 3 has six
24-frame clips from two different recording groups. Check it against all
seven previous packs by recording/day and vehicle, frame ID, JPEG and
decoded RGB identities. Follow the existing spatial preparation and
track-truth sidecar procedures with the new preset:

    python -m scripts.prepare_spatial_sample \
      --sources scripts/spatial-temporal-appearance-holdout-sources.json \
      --metadata-archive /metadata/openscene_metadata_mini.tgz \
      --camera-archive 3=/archive/openscene_sensor_mini_camera_3.tgz \
      --root /outputs/appearance-holdout

    python -m scripts.prepare_spatial_track_truth \
      --root /outputs/appearance-holdout --role appearance_holdout \
      --metadata-archive /metadata/openscene_metadata_mini.tgz \
      --output /truth/appearance_holdout.json

Use the established alignment CPU image without network. Mount archives,
images, prepared captures and weights read-only. Put output outside all inputs.

1. **scripts.capture_spatial_appearance** reopens the previous video and
   tracking captures by their manifests' byte digests and reuses all their
   detections. It restores the exact same camera checkpoint and thresholds.
   Detect only the 144 new frames. Compute descriptors on all 720 frames
   without opening target files; save the printed capture digest.
2. **scripts.run_spatial_appearance** takes the same pack paths and previous
   captures plus that appearance capture and the five truth sidecars with
   byte digests. It computes dev metrics and persists the dev gate before
   first using new targets. It needs no model weights. Both commands expose
   all flags through --help. Neither modifies earlier artifacts.
3. Open report.html for tables and appearance.html for the two-pane player.
   All 144 holdout frames are in the local player. Frame data, vectors, and
   truth stay local, not in Git.

## What is compared

The experiment uses the ordinary zone baseline, strict motion association,
and RGB histogram, pooled DINO and DINO minus local surrounding context
on that same motion recipe. Each profile has a paired control that
permutes its available vectors within each individual frame using a stable
sample-ID-derived ordering. Missing slots never change; singleton vectors
or some other values may stay put. This is an attribution control, **not**
a guaranteed null model. The permutation uses no labels or future frames.

The visual weight 0.4 changes cost, never eligibility: geometry admits a
candidate first and a frame with any missing candidate vectors falls back
to the original geometry. It uses the observed vector only and cannot
generate a box across a gap or absent slot. Descriptor protocol and
observations are authenticated by content and byte hashes.

Preselect only on dev. The natural profile must gain at least .005 IDF1
against both real baselines with no new wrong links or switches, also
gain .005 against its own shuffle on real changed slots. Check only that
same candidate on the unseen clips, not the alternatives. Use the existing
frame IoU assignment and diagnostic fixed-match IDF1, not official MOT.

Separately, for neighboring frames with stable target IDs and a nearby
geometric alternative, compare the same annotated pair-and-distractor
set with real and permuted descriptors. Ties are not successes; zero
eligible queries is not evidence either way. This pair test diagnoses
signal without selecting a recipe or changing the tracker.

One projected object per image quarter is not dense tracking coverage.
None of these results enable physical vehicle decisions.
