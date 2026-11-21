# OpenScene LOC alignment pilot

This research path is separate from the existing LTF/Qwen/Safety evaluator.
It uses a small recorded-camera task to check whether the trainable core can
learn from real pixels and labels, not to reproduce Reason2Drive or evaluate
driving safety. No vehicle control or runtime veto is connected.

## Data boundary

The pinned [OpenScene mini metadata and camera archives](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1)
contain two recordings in camera shard 16. We use 64 uniformly sampled current
front-camera frames from October for optimization and 32 from June for a
recording-disjoint holdout. The count, recording roles and uniform sampling
rule are fixed before model evaluation. This is not the official NAVSIM test
split. Frames within a recording remain correlated.

Each item is one processed current RGB image at 224 × 224 and one task target.
OpenScene associates images with a LiDAR reference frame but does not retain
the precise image capture timestamp; records do not imply exact synchronization.
Use the existing same-K rectification and full-frame resize (no center crop).
The resized calibration follows the RGB pixel-center transform.

The [OpenScene producer](https://github.com/OpenDriveLab/OpenScene/blob/72860746787a67946bef07aa1f78bbbc6b20e445/DriveEngine/process_data/create_openscene_metadata.py)
stores annotated boxes in ego coordinates ordered `x,y,z,length,width,height,yaw`.
We report the selected box using the project's width-before-length layout.
Project the eight oriented cuboid corners through camera calibration. Eligibility
is a vehicle whose front-image box projects entirely in bounds, whose projected
center is in the central horizontal half and whose horizontal ego range is
at most 40 m. The positive target is the closest eligible annotation; the tie
rule is source annotation order. Positive target rectangles are projections,
not human-drawn visible-pixel boxes. Negative means **no eligible annotation**,
not no object, no obstacle, or known empty road. There is no MOT or VQA task.

The prep reads only allowlisted JPEG members and verified metadata from pinned
archives. Metadata is SHA256-checked before deserialization. Prepared images and
supervision are separate hashed artifacts; a train reader does not load holdout
targets. The training side sees labels only for objectives, never as encoder
inputs, ROI coordinates, prompt content or supplied backbone features.

Use the data for the terms of the [nuPlan dataset license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE).
Do not commit media, annotations, cached weights or learned weights derived
from the recordings. Workflow artifacts for the public repo use synthetic
test inputs only. This LOC slice does not replace the official
[Reason2Drive](https://github.com/LogosIntelligenceLab/Reason2Drive) dataset.

## Local preparation

Run in the pinned Linux CPU environment with the already prepared archives
mounted read-only and a separate output path:

```sh
python -m scripts.prepare_alignment_sample   --root /output/loc-slice   --metadata-archive /cache/openscene_metadata_mini.tgz   --camera-archive /cache/openscene_sensor_mini_camera_16.tgz
```

`AlignmentDataset(root, "train" | "holdout")` and `collate_alignment` expose
images `[B,1,3,224,224]` in [0,1], presence `[B]`, 3D ego boxes `[B,7]`
and normalized projected 2D boxes `[B,4]`. Missing boxes are masked targets,
not zeros. The backbone owns RGB normalization; query priors are independent
of these boxes.

## Local pretrained backbones

The pilot uses [DINOv2-small](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
and [Pythia-70M-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped/tree/c92722694b14c67e3f8ed3bf164bb2456449e720)
with pinned immutable revisions. No pretrained vendor files are copied into the
repository. The preparation command fetches only the exact allowlist, with SHA256
verification; the loader rejects other file lists and requires safetensors.

```sh
# Networked preparation (256,395,567 bytes on a fresh setup).
PYTHONPATH=src python -m scripts.prepare_alignment_models --root /path/to/alignment-backbones
# The existing CPU image is the base (`docker build -t hybrid-drive:cpu .`).
docker build -f Dockerfile.alignment -t hybrid-drive:alignment .
```

Run the model smoke with the backbone root mounted read-only at `/models`,
`--network none`, and `ALIGNMENT_ASSET_ROOT=/models`:
`python -m pytest -q tests/integration/pretrained_alignment_smoke.py`.
The regular test environment and its checkpoints do not change. The distinct
image installs hash-pinned Transformers/tokenizers dependencies. It fetches
no model on import, testing or training.

DINO gets a full rectified RGB frame, uses pinned ImageNet normalization and
returns the 16 × 16 patch grid without the CLS token. This is not the classifier
pipeline with center crop. Pythia stays in eval mode with frozen weights but
retains differentiation through the supplied prefix. We use existing single
token IDs for ` LOC` and ` MOT` aliases with no vocabulary change. For this pilot
a fixed prompt ends in consumed ` LOC`; its hidden state is the one eligible
event. The not-yet-consumed yes/no answer is a target, never a prompt input.

## Paired optimization and reporting

Only the prior tokenizer, query projector and instructed decoder update.
Four fixed 2 x 2 image-grid priors carry normalized image-center coordinates
and sizes, not target boxes or pseudo-3D detections. This input is identical
for every frame. Targets supervise a LOC presence bit and a projected 2D/ego
3D box for positives. The fixed prompt has a single consumed LOC event; the
next-token yes/no label is never consumed. There is no MOT supervision.

The zero-feature control uses the same initializer, train order, optimizer,
step count, fixed priors, prompt and labels. Set all visual feature values to
zero in both its training and evaluation, rather than blinding a model only
at evaluation time. The data are split by drive before evaluation; evaluate
holdout at initial and final weights, never select checkpoints or adjust the
recipe using holdout results. Loss is aggregated using supervised counts per
term. Box metrics use all positive labels irrespective of presence prediction.
Use balanced accuracy and its class denominators; yes/no is a forced choice
between token logits, not text generation quality.

In the alignment image with verified `/data` and `/models` mounted read-only,
and `--network none`:

```sh
python -m scripts.run_alignment_pilot   --data-root /data --model-root /models --output-root /results   --revision '<exact git commit SHA>' --epochs 2 --batch-size 4 --seed 0
```

The run saves counts, gradients, per-arm losses and metrics, progress traces,
a standalone HTML report and separate weight-only safetensors. It checks
that the two initial core states match, that CE alone reaches the trainable
bridge, that both frozen backbones stay unchanged, and that the restored core
exactly reproduces a held-out output in the same runtime. Failed runs retain
a diagnostic and completed step records. Do not upload weights fitted on the
restricted data. This is one bounded LOC alignment pilot, not either full
training stage or a new default backend.
