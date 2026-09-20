# Lightweight calibrated-ToF depth refinement

This project predicts a dense 480×640 metric depth map from RGB and an 8×8
VL53L5-style ToF measurement. The input pipeline uses the per-frame calibrated
ToF rectangles stored in ZJU-L5 instead of stretching the 8×8 grid over the
entire camera image.

## RGB/ToF alignment

The dataset supplies the RGB alignment for every ToF zone. For zone `i`,
`hist_data[i]` contains its mean depth and standard deviation, `mask[i]`
contains its validity, and `fr[i] = [top, left, bottom, right]` gives its
calibrated footprint directly in RGB pixel coordinates. The rectangle uses
half-open indexing: `rgb[top:bottom, left:right]`.

The pipeline clips each rectangle to the RGB frame and converts it to
`[mean, std, valid, center_y, center_x, height, width]`. The four geometry
values are normalized by the RGB height and width. At the 1/16 bottleneck,
every RGB location queries all valid ToF tokens; a smooth distance-to-rectangle
bias favors zones whose calibrated footprint is near that RGB location. The
model therefore uses the supplied calibration rather than assuming that the
64 zones form uniform blocks across the RGB image.

See [RGB and calibrated-ToF alignment](docs/tof_rgb_alignment.md) for the
coordinate equations, an example, augmentation rules, and implementation
details.

## Model

Each valid ToF zone is retained as one compact conditioning token containing:

- metric mean depth and distribution standard deviation;
- validity;
- normalized calibrated rectangle center and size.

The model also derives the global mean of valid sensor depths as a scene-scale
anchor. A four-level RGB encoder runs at 1/2 through 1/16 resolution. Local ToF
features enter once through two-head, 16-dimensional cross-attention at the
1/16 bottleneck. RGB features query the 64 ToF tokens, while a smooth geometric
bias favors rectangles near each query without forbidding global context. A
single decoder uses additive RGB skip connections, and the full-resolution head
never receives the rectangular ToF raster. This prevents calibrated zone
boundaries from being copied directly into the prediction.

![Lightweight calibrated-ToF architecture](docs/depth_refinement_architecture.png)

Regenerate the PNG and editable SVG with `python draw_architecture.py`.

    RGB → separable encoder (1/2–1/16) ─┐ queries
                                        ├→ geometric cross-attention
    64 ToF mean/std/valid/box tokens ───┘ keys + values
                                                  │
                                    additive RGB-skip decoder
                                                  │
                                    full-resolution RGB refine
                                                  │
                                         dense metric depth

With the default width of 32, the model has **125,925 parameters** and
approximately **1.40G convolution/attention MACs** at 640×480. The replaced
coarse gate used 33,664 parameters by itself; the cross-attention conditioner
uses only 4,579. The previous nested model had 2.58M parameters and
approximately 116G convolution MACs by the same counting method.

Legacy callers may still pass a one-channel low-resolution depth tensor to the
model. The model then uses validity-normalized interpolation, preventing zero
invalid zones from diluting nearby measurements. Training uses the calibrated
64-zone tokens; the three-channel raster remains available for the global scale
anchor and visualization.

## Setup

    cd /home/himwong/Desktop/depth_refine
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Inspect the configuration, parameter count, and one real batch:

    python main.py
    python main.py --para-summary

The expected training tensors are:

| Key | Shape | Meaning |
| --- | --- | --- |
| image | B×3×480×640 | normalized RGB |
| sparse_depth | B×1×8×8 | raw ToF mean for display/compatibility |
| sparse_valid_mask | B×1×8×8 | raw zone validity |
| tof_features | B×3×480×640 | calibrated mean/std/validity |
| tof_tokens | B×64×7 | mean/std/validity plus normalized rectangle geometry |
| target_depth | B×1×480×640 | metric ground truth |
| target_valid_mask | B×1×480×640 | valid ground-truth pixels |

## Data split

The manifest defines scene-disjoint splits:

| Split | Samples | Scenes |
| --- | ---: | --- |
| Train | 707 | cafe1, cafe2, classroom, dinner_room1, lab1, lab2, leisure_area, library1, supermarket, teaching_region |
| Validation | 101 | library2 |
| Test | 202 | dinner_room2, dorm, showroom, theater |

Training augmentation applies horizontal flips to RGB, calibrated ToF, and
ground truth together. Gamma, brightness, and color augmentation affect RGB
only.

## Training

The default objective is masked MSE because model selection and final reporting
use metric RMSE. Start a new lightweight run with:

    python train.py

Checkpoints are written to checkpoints_attention_v3/ and TensorBoard events to
runs/depth_refinement_attention_v3/. These checkpoints are intentionally
separate because the conditional-attention architecture is incompatible with
the earlier coarse-fusion and direct-blending models.

TensorBoard:

    tensorboard --logdir runs/depth_refinement_attention_v3

Training records total validation RMSE and RMSE for 0–2 m, 2–4 m, 4–6 m, and
6+ m. The far-depth curve is useful because a small number of distant pixels can
dominate aggregate squared error.

For a pipeline smoke test, temporarily set device to cpu, epochs to 1, and both
max_train_batches and max_validation_batches to 1 in config.yml.

Resume a lightweight checkpoint by setting a larger total epoch count:

    python train.py --resume checkpoints_attention_v3/depth_refinement_epoch_005.pt --epochs 60

## Evaluation

Evaluate the best checkpoint on the held-out test scenes:

    python eval.py --checkpoint checkpoints_attention_v3/best.pt

The evaluator reports masked MAE, total RMSE, and the four depth-bin RMSE
values. Save RGB, calibrated ToF, prediction, and ground-truth panels with:

    python eval.py --checkpoint checkpoints_attention_v3/best.pt --visualize

All model, data, training, and evaluation settings are in config.yml.
