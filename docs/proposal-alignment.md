# Proposal priors for four-band LOC alignment

This offline path connects the existing vision and language pieces without
changing the Fast–Slow–Safety runtime. DINOv2 encodes a single current rectified
front frame. The frozen spatial head proposes one 2D image box per quarter.
PriorTokenizer pools those areas from the frame's feature map and embeds
their normalized image center and size. Q-Former projects the resulting
tokens together with the same whole-frame features into a Pythia prefix.
The instructed decoder reads the states after the four fixed LOC markers.

The source admits only the **image features, seed 0** arm of the earlier
spatial run. It binds a caller-supplied result byte SHA256 to the saved
manifest, dev calibration, exact DINO receipt, data manifest and checkpoint.
Weight bytes, metadata and trained tensor state must agree. Its existing
dev threshold is validated as provenance, not used to discard boxes.
The source always returns four normalized, finite left-to-right RoIs. Nothing
claims that a proposal is a ground-truth object or safe driving decision.

The comparison keeps four RoIs and the whole-frame input in both conditions.
The control uses full-height fixed quarters; the treatment uses the frozen
predicted boxes. Neither directly copies a proposal into the decoded box.
Quarter identity comes from the four LOC positions. Q-Former attends to the
common visual and ROI token bank; it does not assign one attention query
exclusively to one proposal. The bridge checks ROI quarter order at its input.

The causal prompt includes four consumed LOC markers and no answers. Text
cross-entropy compares the model's next-token logits to the yes/no labels
at those fixed positions. This is a forced-token diagnostic, not generated
VQA. The instructed decoder's presence scores and 2D-box GIoU use the aligned
four-quarter targets; 3D and MOT are not supervised. The forward path
receives images/features and RoIs, never target boxes or labels.

Use the same pinned dataset and alignment CPU image as the earlier
spatial and Pythia experiments. No new pretrained assets or camera data
are required. Local recordings, raw targets and trained weights do not
belong in this repository.

## Frozen comparison

The prepared pack keeps the original 96 train, 48 dev and 64 reference frames.
The two reference recordings were already evaluated in the earlier spatial
benchmark; this run is a paired mechanism study on a reused holdout.
Seed 0 and seed 1 each use an identical initialization and batch order inside
their fixed/proposal pair. The recorded recipe is AdamW at 0.002, three epochs
and batch size four. DINO, Pythia and the proposal head are not optimized.
Text-only CE gradients are checked for the tokenizer and projector.

Calibrate decoder presence and the post-LOC yes-minus-no score separately
on dev. Save the four safetensors checkpoints and both thresholds per run,
then hash and re-read calibration before constructing the reference reader.
Compare both types of score and projected-box IoU on the reference split.
The shuffled control changes only the four proposals, rotating frame
association within each recording. Whole-frame features, targets and dev
thresholds stay the same. Do not use the reference scores to select the
proposal-producing checkpoint or modify the recipe.

Run the comparison in the pinned alignment CPU image with verified read-only
inputs and `--network none`:

```sh
python -m scripts.run_proposal_alignment --data-root /data --model-root /models   --prior-run-root /prior --prior-result-sha256 '<trusted result.json SHA256>'   --output-root /results --revision '<exact git commit SHA>'   --epochs 3 --batch-size 4 --seeds 0 1
```

The summary records both metric channels, per-recording rows, paired deltas
and gradients. Retain the report and calibration receipt alongside the saved
states. This does not implement FSDrive priors or evaluate the 5 ms timeline
of a runtime bridge.


## Fresh holdout with isolated visual memories

Prepare only the separate holdout in the pinned CPU image. Reuse the original
spatial train/dev pack; do not expose the old reference as a final result.
Use the official OpenScene mini camera shard 4 and the pinned metadata archive:

    python -m scripts.prepare_spatial_sample \
      --root /results/spatial-loc-fresh \
      --sources scripts/spatial-loc-evaluation-sources.json \
      --metadata-archive /sources/openscene_metadata_mini.tgz \
      --camera-archive 4=/sources/openscene_sensor_mini_camera_4.tgz

The new pack is bound to two recordings, 32 frames each. The experiment
compares its recording, vehicle/date, and frame/JPEG/RGB identities against
**all** original roles. The validated holdout manifest hash and all dev
thresholds are saved before the final label reader is constructed. The
label reader checks that exact manifest hash before reading target bytes.

    python -m scripts.run_proposal_alignment \
      --data-root /data --fresh-holdout-root /fresh --model-root /models \
      --prior-run-root /prior --prior-result-sha256 '<trusted SHA256>' \
      --output-root /results --revision '<tested git SHA>' \
      --epochs 6 --batch-size 4 --seeds 0 1

Both arms use precisely four pooled tokens in **both** Q-Former and the
decoder. The frame-only control pools four fixed full-height tiles, covering
the whole frame without predicted coordinates. The learned-region condition
pools four predicted ROIs from the same DINO patch map. Neither arm passes
the unpooled grid downstream. DINO and the proposal generator still use
whole-frame context upstream. Fixed initialization, batch order, optimizer,
bridge dimensions and input token count match within each seed.

We keep the already-trained seed-0 spatial head as a direct no-LLM reference
and use its **original** dev threshold without calibration on the two new
recordings. The direct control shuffles the deterministic spatial outputs
within each recording while keeping labels fixed; this is equivalent to
shuffling source images before the frozen single-frame pipeline. For the
bridge control only proposal coordinates are shifted to another frame; ROI
tokens still pool from the current frame.

Open the generated report for aggregate metrics and the linked local gallery
for 12 uniformly chosen frames. The gallery always shows the first recipe
seed; do not select it with final scores. It draws the reference projections
and predictions above each saved decoder threshold. Frames, weights and
per-frame annotations remain local. Neither metric nor visualization
changes the driving agent or passes a production perception requirement.
