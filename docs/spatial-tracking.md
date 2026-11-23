# Identity continuity on front-camera video

This is attribution, not another detector. The previous calibrated camera head
remains frozen. Per frame, only boxes that the existing policy marked PRESENT
receive a provisional identity. Identity never changes scores, decisions,
box coordinates or the number of boxes. The tracker does not draw anything
from history and does not estimate physical car speed or steering.

## Targets and split

Reuse the 432 already captured frames (three known video packs) for development.
Prepare the pinned new tracking_holdout clip pack on camera shard 23: two new
recordings, six clips with 24 frames each. Check the recording, day/vehicle,
frame-token, JPEG and RGB identities against the six previous packs.

The sidecar generator uses the verified OpenScene metadata archive and current
pack manifest. It recomputes the camera projection and verifies the actual
target box and frame alignment. Only the selected vehicle receives a tracking
alias. The generator uses stable source track tokens, not per-observation
instance tokens. Original tokens stay out of the sidecar and the player. The
ordinary box-target files, model and prior observations stay unmodified.

In the same alignment CPU image with no network and read-only inputs:

    python -m scripts.prepare_spatial_sample \
      --sources scripts/spatial-temporal-tracking-holdout-sources.json \
      --metadata-archive /old/openscene_metadata_mini.tgz \
      --camera-archive 23=/new/openscene_sensor_mini_camera_23.tgz \
      --root /output/tracking-holdout

For each old video role and the new tracking_holdout role, use:

    python -m scripts.prepare_spatial_track_truth \
      --role tracking_holdout --root /output/tracking-holdout \
      --metadata-archive /old/openscene_metadata_mini.tgz \
      --output /truth/tracking_holdout.json

Store the four printed sidecar SHA-256 digests. Making a pack and its evaluation
sidecar is not an instruction to use its labels for training or tuning.

## Capture and evaluate

- **scripts.capture_spatial_tracking** reopens the previous video's capture
  (same six verified previous pack roots) and authenticates the frozen model
  against its model hash. It captures only the 144 new frames, without opening
  box targets or sidecars. Save the printed tracking capture SHA-256.
- **scripts.run_spatial_tracking** loads both captures and four sidecars
  with explicit byte digests. It evaluates and freezes the development choice
  before opening new track labels. It stores counts and a standalone
  video comparison, without putting media or per-frame IDs in Git.
- Both scripts offer --help. Use distinct output directories outside all input
  roots. Verification does not require a physical camera or a simulator.

## Six modes and measures

The baseline gives an ID to consecutive displayed boxes in the same image
quarter. The alternatives use one-to-one IoU or constant-velocity image-space
matching with conservative geometric gates, optional continuation after one
**observed** unknown frame, or no match when the alternatives are ambiguous.
A skipped capture, explicit absence, or clip boundary cannot inject a new
visible object. No ID is assumed to connect the same scene across separate clips.

Compare exactly the same detections and target sets. Assign frame matches by
geometry first; tracking IDs never influence which boxes match. The fixed-match
IDF1 diagnostic computes how much of those matches can be assigned to an
actual identity consistently within a clip. It is not an official MOTChallenge
submission. Report wrong inspected links, adjacent label-ID opportunities,
switches, and links across quarters as counts as well. Selection requires
an improvement in dev IDF1 of 0.005 without more wrong links or switches.
On final clips check only the dev-selected mode against the baseline; alternatives
are diagnostics. One projected eligible box per quarter is not exhaustive
annotation coverage, and an overlay is not a driving decision.
