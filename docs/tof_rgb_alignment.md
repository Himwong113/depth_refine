# RGB and calibrated-ToF alignment

Each sample contains one RGB image and 64 ToF zones. The four arrays are joined
by zone index:

| Source | Shape | Meaning for zone `i` |
| --- | --- | --- |
| `rgb` | `H × W × 3` | RGB coordinate frame |
| `hist_data` | `64 × 2` | `[mean_depth_m, standard_deviation_m]` |
| `mask` | `64` | sensor validity |
| `fr` | `64 × 4` | `[top, left, bottom, right]` in RGB pixels |

The calibration is already stored in `fr`. The loader does not estimate camera
extrinsics or infer the position from the zone's row and column in the 8×8
sensor grid.

```text
hist_data[i] ─ mean and uncertainty ─┐
mask[i]      ─ validity ─────────────┼─ zone i ─ calibrated token ─┐
fr[i]        ─ RGB rectangle ────────┘                              │
                                                                  ▼
RGB image ─ RGB encoder ─ spatial queries ─────────────── cross-attention
```

## Rectangle convention

For `fr[i] = [top, left, bottom, right]`, the loader first clips the rectangle
to the RGB image:

```text
y0 = clip(top,    0, H)       x0 = clip(left,  0, W)
y1 = clip(bottom, 0, H)       x1 = clip(right, 0, W)
```

The footprint is the half-open region:

```text
rgb[y0:y1, x0:x1]
```

Thus, `bottom` and `right` are excluded. A zone is usable only when all of the
following hold:

- `mask[i]` is true;
- mean depth is finite and greater than zero;
- standard deviation is finite and non-negative;
- `y1 > y0` and `x1 > x0`.

Invalid zones remain present in the 64-token tensor but are masked out by
attention.

## Token geometry

The clipped RGB rectangle is converted to normalized geometry:

```text
center_y = (y0 + y1) / (2H)
center_x = (x0 + x1) / (2W)
height   = (y1 - y0) / H
width    = (x1 - x0) / W
```

The final token is:

```text
[mean_depth_m, std_m, valid, center_y, center_x, height, width]
```

For example, with a `480 × 640` RGB image and:

```text
fr[i] = [120, 160, 240, 320]
```

the zone covers rows `120..239` and columns `160..319`. Its normalized geometry
is:

```text
center_y = 0.375    center_x = 0.375
height   = 0.250    width    = 0.250
```

The implementation is in `build_calibrated_tof_tokens` and
`rasterize_calibrated_tof` in [data/zjul5.py](../data/zjul5.py).

## How attention preserves the alignment

The RGB encoder produces a feature map at 1/16 resolution. A feature at
bottleneck row `r` and column `c` is assigned the normalized RGB position:

```text
query_y = (r + 0.5) / bottleneck_height
query_x = (c + 0.5) / bottleneck_width
```

For every query and every valid ToF token, the model measures the query's
distance outside the calibrated rectangle:

```text
dy = max(abs(query_y - center_y) - height / 2, 0)
dx = max(abs(query_x - center_x) - width  / 2, 0)
```

The cross-attention score combines learned RGB/ToF compatibility with this
geometry:

```text
score = QKᵀ / sqrt(head_dim) - softplus(beta) * (dx² + dy²)
```

A query inside a zone has zero geometric penalty. The penalty grows smoothly
outside the rectangle, so nearby zones are preferred while content attention
can still use a more distant zone. Invalid zones receive no attention weight,
and a zero-valued null token lets the network ignore ToF when no measurement is
useful.

The resulting ToF context is added only at the coarse RGB bottleneck. The
piecewise-constant raster is not copied into the full-resolution decoder, which
avoids creating hard rectangle boundaries in the predicted depth.

## Raster and token roles

The loader produces two aligned representations:

| Representation | Shape | Use |
| --- | --- | --- |
| `tof_features` | `3 × H × W` | visualization and global metric-scale anchor |
| `tof_tokens` | `64 × 7` | keys and values for geometry-aware attention |

For the raster, a valid zone writes its mean, standard deviation, and validity
into `[:, y0:y1, x0:x1]`. The network does not use this local raster as a final
depth shortcut.

## Augmentation and resizing

A horizontal flip must transform every aligned input together. The loader:

- flips RGB, target depth, and the ToF raster along the width axis;
- updates each token with `center_x = 1 - center_x`;
- keeps token width, vertical center, and height unchanged.

Pure image resizing preserves normalized token geometry. Cropping, rotation,
or perspective augmentation would require applying the same transform to every
ToF rectangle; these transforms are therefore not used by the current loader.

When adding another dataset, confirm that its rectangle order and coordinate
convention match `[top, left, bottom, right]`. Swapping `x/y`, treating
`bottom/right` as inclusive, or stretching the raw 8×8 grid uniformly will
misalign the sensor condition from the RGB features.
