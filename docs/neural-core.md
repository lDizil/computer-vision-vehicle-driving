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

## Integration boundary

No detector, encoder or 3D object producer is invented by this module. In
particular, FSDrive does not currently expose a verified object-query API.
Supplying synthetic features verifies neural mathematics, not perception
quality or shared-feature alignment. The existing LTF/Qwen/Safety path is not
silently switched to newly initialized weights.
