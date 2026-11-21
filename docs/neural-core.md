# Trainable reasoning core

These project-owned PyTorch modules implement explicit interfaces inspired by
[Reason2Drive, section 4 and Appendix B](https://arxiv.org/html/2312.03661v3).
The public upstream does not provide a released model implementation/checkpoint.
These classes do not claim checkpoint compatibility or reproduced paper results.
They are separate from the existing image-input advisory runtime and Safety.

## Prior tokenizer

`PriorTokenizer(image_channels, geometry_dim=7, token_dim=768, hidden_dim=768)`
accepts:

| Input | Shape / meaning |
|---|---|
| `feature_maps` | `[B,V,C,H,W]` float32/float64 from an external real encoder |
| `boxes_xyxy` | `[B,N,4]`, normalized positive-area RoIs in the matching preprocessed image |
| `image_indices` | `[B,N]` long, ordered camera/frame view index |
| `geometry` | `[B,N,G]`, separate spatial/motion descriptors; not inferred from a 2D RoI |
| `valid_mask` | `[B,N]` bool, True for actual priors |
| `image_valid_mask` | optional `[B,V]` bool; a prior cannot reference a missing view |

The default geometry order is `[x,y,z,width,length,height,yaw]`, in the
snapshot ego frame x-forward/y-left/z-up, metres/radians. This is our contract,
not a dimension/order guaranteed by the paper. A different geometry layout
must be declared by its dimension and understood by the caller; optional
`geometry_feature_names` records its ordered names. Normalization of those
features belongs to the producer, not an inferred operation in the tokenizer.

The tensor feature-map resolution need not equal the original image resolution.
We multiply normalized x coordinates by W and y coordinates by H independently,
then use the actual `torchvision.ops.roi_align` with `7 x 7`, `spatial_scale=1`,
`aligned=True`, `sampling_ratio=2`. Wrong boxes are rejected, not clamped or
reordered. Calibration/crop transforms belong to the producing encoder adapter.

Our architecture uses learned linear spatial pooling `49 -> 1`, a geometry
embedding, concatenation and a two-linear-layer fusion MLP. Output is
`PriorTokens(tokens=[B,N,768], valid_mask=[B,N])` by default. Width, hidden
size, pooling details and geometry schema are explicit project choices.

Only active priors reach RoIAlign and the trainable chain. Padding may contain
unknown values and remains zero at the output; empty N is valid and backward
safe. Tensor references follow ordinary PyTorch mutability/autograd semantics,
not the immutable NumPy inference-contract semantics.

## VisionTransformer feature adapter

`ViTPatchEncoder` reuses the public torchvision FX feature extractor on a
supplied `VisionTransformer`. It exposes `encoder.ln` tokens, removes exactly
one CLS token and restores a known patch grid from the encoder image/patch
sizes. Input is `[B,V,3,S,S]`; output is `[B,V,hidden_dim,S/P,S/P]`.
There is no sqrt-based grid inference or automatic image preprocessing.

The wrapper shares the supplied encoder parameters and does not download,
freeze, detach, resize, normalize or set inference mode. The caller owns those
choices. Tests use an actual tiny torchvision ViT initialized from scratch,
including ViT -> PriorTokenizer(768) backward. This verifies the integration
boundary, not pretrained EVA-CLIP features, perception quality or alignment.

## Integration boundary

No detector, encoder or 3D object producer is invented by this module. In
particular, FSDrive does not currently expose a verified object-query API.
Supplying synthetic features verifies neural mathematics, not perception
quality or shared-feature alignment. The existing LTF/Qwen/Safety path is not
silently switched to newly initialized weights.

## Query projection and instructed decoder

`QFormerProjector(visual_dim, llm_dim, prior_dim=768, ...)` projects external
visual tokens `[B,M,C]` and optional prior tokens `[B,N,768]` into fixed learned
queries `[B,Q,H_llm]`. Boolean masks use True for valid tokens. We sanitize
padding before projections/attention and require actual visual evidence in
every batch row. Empty priors are valid; all-masked visual rows are not.

This Q-former-like implementation uses `nn.TransformerDecoder`, learned query
and visual/prior type embeddings. It does not load LAVIS/InstructBLIP weights.
Query count/layers/width are explicit project configuration. Spatial/view
position information must already be represented by the encoder's features.
A compatible external LLM adapter owns prefix injection and tokenization.

`InstructedVisionDecoder` receives visual memory, consumed token IDs, matching
last-layer hidden states and an `answer_mask`. `<LOC>` / `<MOT>` IDs are supplied
by that tokenizer adapter. We take `h_t` after consuming the special token, not
`h_(t-1)` that predicted it. Prompt/padding occurrences are not answer events.
Repeated events are retained in batch/token order and are never truncated.
Each event is decoded independently: later generated events cannot influence
an earlier prediction. Its LLM hidden state already contains causal context.

`PerceptionOutput` is a flat event table:

| Field | Shape / meaning |
|---|---|
| `batch_indices`, `token_positions` | `[E]` long, exact event identity |
| `event_kinds` | `[E]` long, LOC=0 / MOT=1 |
| `validity_logits` | `[E]`, raw classification logits |
| `boxes_3d` | `[E,7]`, raw x/y/z/width/length/height/yaw regression, ego metres/radians |
| `boxes_2d` | `[E,4]`, auxiliary normalized-image xyxy; positive extents, corners may be outside the image |
| `motion_xy` | `[E,T,2]`, ego metres at dt through T*dt; default T=8, dt=0.5 s |

The auxiliary 2D box is independently predicted, not a calibrated projection
of the 3D box. It supports the project's 2D GIoU loss without pretending that
axis-aligned 2D overlap measures oriented 3D overlap. Raw untrained box/motion
outputs are not physically validated detections or driving trajectories.
Empty E is backward-safe. Tensors keep ordinary PyTorch ownership semantics.

## Project loss and synthetic optimization

`Reason2DriveLoss` follows our project ticket recipe:

```
L = CE_text + lambda_per * (BCE_validity + box_weight * SmoothL1_3D
                           + giou_weight * GIoU_2D + motion_weight * SmoothL1_motion)
```

This is not claimed to reproduce the paper's BCE/MSE recipe. The oriented
7D box and auxiliary image-space 2D box are different heads. GIoU uses the
latter only. The 3D yaw residual wraps to its shortest angular arc before
SmoothL1; target box dimensions are positive. Motion targets follow the
configured decoder sampling in the same ego frame.

Text logits and labels must already be aligned by the LLM adapter; the loss
cannot infer decoder-only versus encoder-decoder shifting. Event targets
match exact batch/token positions and LOC/MOT kinds. Each term averages its
own supervised elements. Negative validity, unknown label and padding are
separate: regression uses positive events of the relevant task and only
known coordinates/future steps. Unknown future is not a zero target.
All-ignore/empty components remain differentiable zeros without reading
padded NaNs. Nonfinite supervised inputs and mismatched event identity fail.

`run_neural_core_smoke` executes real CPU optimizer steps through a frozen
tiny torchvision ViT and a frozen tiny causal Transformer fixture. They are
newly initialized, not pretrained encoders or LLMs. The core tokenizer,
projector and instructed decoder update; frozen weights stay bitwise equal.
We do not place the frozen LLM forward under `no_grad`: input-prefix gradients
must still pass through it. A regression test detects accidental detachment.

The experiment records per-component losses/gradient norms and a self-contained
plot. It is a synthetic mathematical smoke, not real-data alignment, held-out
quality, a full Reason2Drive checkpoint, Stage 1 completion or better driving.

Neural weight artifacts use bounded safetensors, explicit expected SHA256 and
experiment metadata, strict keys/dtypes/shapes and prevalidation of every
module before copying state. They contain core weights only, not optimizer
resume state or a dynamically imported model factory. No pickle is loaded.
