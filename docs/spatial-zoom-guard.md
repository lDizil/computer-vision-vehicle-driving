# Guarded zoom on new recordings

A close-up can move an existing box toward the object or away from it. This
experiment checks when it is better to keep the original box or apply only
half of a crop-based correction. It does not change the detector, presence
thresholds, published refiner, tracking rules, or runtime control. The crop
resamples the same 224×224 image; it adds no sensor detail.

## Data and roles

| Use | Pinned packs | Frames |
|---|---|---:|
| Fit / cross-fit | Temporal `development`, `stress`, and the previously observed `holdout` | 432 |
| Select on dev | Previously observed `tracking_holdout` and `appearance_holdout` | 288 |
| Compare once | New `zoom_gate_holdout` from camera shard 1 | 144 |

The new pack contains two recording/day-and-vehicle groups and six 24-frame
contiguous clips. Admission checks the eight exact earlier packs bound to the
camera training and capture lineage: recording and date/vehicle group, log and
frame tokens, JPEG member and digest, and decoded RGB digest. The source camera,
projection and target rules stay identical. The new role never becomes a fitting
or recipe-selection input. Keep source archives, captures, weights and truth
local. Do not publish them or the image-bearing viewer to Git.

Prepare the new pack and its separate track-truth sidecar with the existing
alignment CPU image, no network, read-only metadata and camera archives, and
a writable destination outside all earlier inputs:

```bash
python -m scripts.prepare_spatial_sample \
  --sources scripts/spatial-temporal-zoom-gate-holdout-sources.json \
  --metadata-archive "$METADATA_ARCHIVE" \
  --camera-archive "1=$CAMERA_SHARD_1" --root "$FRESH_ROOT"
python -m scripts.prepare_spatial_track_truth \
  --root "$FRESH_ROOT" --role zoom_gate_holdout \
  --metadata-archive "$METADATA_ARCHIVE" --output "$FRESH_TRUTH"
```

1. **`python -m scripts.capture_guarded_zoom`** takes the previous data/capture
   paths and manifest digests used for the zoom experiment, `--known-zoom-root`,
   `--known-zoom-sha256`, and `--fresh-root`. Also supply the same model/prior/
   accuracy roots and prior/accuracy result digests. This step reuses all known
   features and detects/encodes only the 144 new frames with the unchanged camera
   weights. It opens no target or track-truth files. Keep the printed SHA-256
   of `fresh-capture.json`.
2. **`python -m scripts.run_guarded_zoom`** takes the same previous data/capture
   paths and digests, known-zoom root/digest and fresh root, plus
   `--fresh-capture-root`, `--fresh-capture-sha256`, and one `--<role>-truth`
   with `--<role>-truth-sha256` for each of the six roles in the table. Use
   hyphens in flags (e.g. `--zoom-gate-holdout-truth`). It does not load the
   pretrained vision encoder or need its downloaded weights. Both commands
   list all flags via `--help`; point `--output-root` outside every input and
   use `--revision` for the exact source commit.
3. Open `report.html` for the dev choice, fresh results, and per-recording
   differences; `guard.html` compares the same frames in a local two-pane
   player. `result.json` contains scores, artifact hashes, and run status.

## Freeze the choice before reading fresh labels

For each of the three fitting folds hold out an entire two-recording pack,
fit the box head on the other two, and predict the held fold. Fit the guard
on those out-of-fold proposals, not on in-sample head predictions. The guard
estimates the gain from moving half or all the way toward the zoom proposal.
Train the final proposal head on all three fitting packs. The five compared
recipes are original, full correction, half correction, learned choice, and
learned choice with an additional training-range check.

Dev selection counts one-to-one frame matches at IoU ≥ 0.5. A recipe must add
at least two matches across both dev packs, with no net drop in either pack
and at most one lost net match per recording. Ties prefer fewer changed boxes,
then lexical recipe name. If no recipe qualifies, keep the original. Save the
choice and the training evidence in `development-gate.json`; reopen that gate,
verify the guard bytes, and restore the bound final checkpoint before the
benchmark opens the new truth sidecar or targets.

On the new clips, evaluate the preselected recipe against the same thresholds.
Failure means no replacement; no other recipe is chosen from the fresh results.
Gained and lost matches count distinct frame-level target transitions even when
the net gain is zero. IDF1 from the unchanged zone and strict-motion tracking
rules is secondary, not a selection signal. The label-aware oracle checks the
upper bound of choosing among the three fixed box positions for every visible
slot under a shared one-to-one assignment. It is not a deployable recipe.
