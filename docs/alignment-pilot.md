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
