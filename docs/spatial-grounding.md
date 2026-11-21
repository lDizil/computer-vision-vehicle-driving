# Spatial LOC evaluation on OpenScene

The spatial experiment is a different task from the two-recording alignment
pilot. It tests image-dependent 2D location, without an LLM or a runtime
veto. No view of the final targets is needed to set the training recipe.

## Fixed source pack

We keep the two already examined camera shard 16 recordings in train only.
The two official additional shards are
[5](https://huggingface.co/datasets/OpenDriveLab/OpenScene/blob/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_5.tgz)
and [6](https://huggingface.co/datasets/OpenDriveLab/OpenScene/blob/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_6.tgz).
Their source names are checked against the pinned publisher map and metadata.
The package uses six unique recordings (96 train / 48 dev / 64 final frames).
Two dev recordings come from the same vehicle and day, and stay in one role;
no full recording or vehicle/day group crosses roles.

Select uniform source-list midpoints before inspecting target outcomes.
The builder verifies the complete archive and each corresponding trusted
metadata member before pickle, reads only allowlisted CAM_F0 images, and saves
rectified RGB at 224 × 224. Each role has its own hashed supervision file.
Train/dev readers do not open final targets. Cross-role duplicate frame IDs,
JPEG digests or decoded RGB digest are rejected, not substituted.

Each frame has four simultaneous queries, over consecutive non-overlapping
horizontal coordinate quarters from left to right. A box belongs to the band
of its projected 2D AABB center computed in float32, not the 3D center ray.
Select the nearest eligible annotated vehicle per band. Eligibility uses the
existing in-frame, depth, range and projected extent constraints. Negative
means no projection-qualified source annotation, not verified absence.

No labels or per-frame provenance enter the encoder. Positive boxes are
calibrated projections, not human annotations of visible pixels. Frames within
one recording are correlated. Use only under the source data terms. Prepared
pixels, target files and learned weights stay outside the repository.

```sh
python -m scripts.prepare_spatial_sample --root /results/spatial-loc   --metadata-archive /sources/openscene_metadata_mini.tgz   --camera-archive 5=/sources/openscene_sensor_mini_camera_5.tgz   --camera-archive 6=/sources/openscene_sensor_mini_camera_6.tgz   --camera-archive 16=/sources/openscene_sensor_mini_camera_16.tgz
```

Run in the pinned CPU image with sources read-only and no network. By default
the CLI reports the frozen identities and sample counts, not target statistics.
