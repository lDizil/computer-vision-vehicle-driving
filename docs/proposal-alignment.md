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
