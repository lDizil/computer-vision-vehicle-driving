# Spatial confidence and box accuracy

The offline spatial head gives one score and one normalized 2D box for each
of four vertical zones. The new policy returns **present**, **absent**, or
**uncertain** and only shows a box when the score justifies a positive result.
The old policy, reproduced exactly, uses the existing cutoff for both
boundaries. The three-way policy never treats abstention as a correct
negative.

Fit two global cutoffs on the original dev (48 frames) and the previously
examined two recordings from the earlier gallery (64 frames). That second
pack is now explicitly development data, not an independent test. On every
development recording constrain false confident presence to at most 20%
of all negatives and false confident absence to at most 10% of all positives
where those classes are available. The boundaries maximize decisions within
these constraints, respecting ties. These limits are a dev operating point,
not a statistical guarantee for unseen recordings.

The residual refiner sees boxes from the frozen spatial head and a small
local neighborhood on the same DINO feature map. It adjusts center and
size while preserving the intended zone and positive bounds. It never changes
the presence score. Train two seeded copies only on the unchanged 96
training frames, and compare dev projection IoU to the unmodified boxes.
Choose a candidate only with at least +0.01 recording-macro dev IoU and no
dev recording regression beyond 0.02. If none qualify, retain the original
geometry. Resolve qualifying ties by seed. DINO and the spatial head are frozen.

The new test is pinned camera shard 19, two other recordings with 32 frames
each. The loader checks identities and JPEG/RGB digests against both older
packs. Before reading any final labels the runner saves/reopens the cutoffs
and selection and reloads both exact checkpoint states. All candidate IoUs
remain available for analysis, even if a candidate was rejected on dev.

Run the CLI in the existing restricted alignment CPU image, with all inputs
read-only and network disabled:

    python -m scripts.prepare_spatial_sample \
      --sources scripts/spatial-loc-policy-evaluation-sources.json \
      --metadata-archive /old/openscene_metadata_mini.tgz \
      --camera-archive 19=/sources/openscene_sensor_mini_camera_19.tgz \
      --root /outputs/policy-holdout

    python -m scripts.run_spatial_accuracy \
      --reference-root /original --calibration-root /old-gallery \
      --evaluation-root /policy-holdout --model-root /models --prior-root /prior \
      --prior-result-sha256 '<trusted spatial result SHA256>' \
      --revision '<tested git SHA>' --output-root /results --epochs 12 --batch-size 8

Report presence precision, recall, coverage and uncertainty together. The IoU
on all positive targets is separate from the count of displayed boxes that
match a positive target at IoU ≥ 0.5. In the local gallery, shaded columns
are uncertainties; boxes are never filled in from reference labels.
No photos, labels or prepared checkpoints go in the repository.
This is one-camera projected annotation evaluation, not driving or safety
validation.
