# Spatial judgments on actual video sequences

This is an offline test of a front camera model across time, not a control
loop. It restores the exact calibrated DINO + spatial head + chosen box
refinement of the preceding accuracy experiment, without training or VLM
inference. The two confidence thresholds do not change.

## Data and boundaries

Each pack has two recordings × three 24-frame clips (144 captures).
Intervals are about 0.5 s from associated source timestamps, not camera
clock or computation latency measurements.

- Development uses shard 4, previously seen material.
- Holdout uses shard 22, two new recordings. Recording and day/vehicle
  identities, frame IDs and source JPEG/decoded RGB hashes must be distinct
  from every previous model pack and the other video roles.
- Stress uses shard 19, previously seen material. Faults are injected only here.

Use the alignment CPU image with no network, read-only inputs and a separate
writable results directory. Example to prepare one of the three packs:

    python -m scripts.prepare_spatial_sample \
      --root /results/holdout \
      --sources scripts/spatial-temporal-holdout-sources.json \
      --metadata-archive /source/openscene_metadata_mini.tgz \
      --camera-archive 22=/source/openscene_sensor_mini_camera_22.tgz

For development and stress change the manifest and shard to 4 or 19.
Keep the original spatial packs for provenance. Do not request target
statistics when preparing the final pack.

## Run and reproduce

1. **scripts.capture_spatial_video** binds the two prior model artifacts to
   their result-file SHA-256 and captures 432 frames without reading target
   files. Pass development, holdout and stress roots, plus reference,
   previous-dev and accuracy-holdout roots for the three prior packs.
   Supply model, prior and accuracy roots, both trusted result digests,
   output root and the tested revision. Save the printed capture hash.
2. **scripts.run_temporal_spatial** takes the same six data pack paths, the
   capture root and its hash, output root and the tested revision.
   It compares nine recipes, freezes and reopens the development choice,
   then opens the test targets. The HTML report and interactive player stay local.
3. Repeat the second command with **--verify-benchmark** pointing to the
   completed run and **--benchmark-sha256** instead of the revision. This
   recomputes all metrics and stress results without model weights or RGB
   reads. It requires the same matrix hash and writes a separate receipt.

Both CLI commands accept --help. Do not put frames, traces, annotations
or model checkpoints in Git.

## What counts as an improvement

- Compare framewise, two/three-frame confirmation, a one-frame hold on
  current associated geometry, score EMA, box EMA and three combinations.
- Measure false confident presence/absence, recall, abstention, changes on
  unchanged targets, detection delay and motion-compensated center jitter.
  Paired jitter uses only edges where both modes showed boxes. A positive at
  the start is not an appearance; a missing frame is not a temporal edge.
- Select on dev only: no more false confident answers or missed appearances,
  ≤2 percentage points of recall loss, ≤0.5 s extra median appearance delay,
  ≤0.01 macro IoU loss and ≤1 lost accurately placed displayed box.
  Require a measured improvement. Check only the preselected variant
  against framewise on holdout; don't shop among test results.
- Test four faults: every fifth frame dropped, equally patterned stale
  delivery, timely delivery jitter and invalid inserts. Dropped and
  late streams must match, and invalid inserts cannot corrupt the remaining
  stream. The benchmark explicitly checks cadence is slower than the fake
  stale delay so it does not simulate impossible delivery order.

The nearest eligible projected annotation is not manual pixel truth.
Overlap association is not object identity tracking. The results do not
authorize enabling a driving or safety decision.
