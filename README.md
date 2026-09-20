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

## Teacher/student v4

The research path is now implemented as two explicit architectures:

- `teacher_v4`: a frozen, externally pretrained Depth Anything V2 Large RGB
  backbone plus trainable ZJU-L5 ToF fusion and metric-depth decoder;
- `student_v4`: a deployable 141,694-parameter network with internal 320×240
  RGB processing, fusion at 1/16 and 1/8, and a narrow full-resolution head.

Both fusion blocks pool RGB appearance inside each calibrated ToF footprint.
One attention head is geometry-biased and one remains global in the student;
the teacher uses two of each. Invalid sensor values are masked and every block
has a learnable null token. The deployment API is:

```python
depth = student(image, tof_tokens)
output = student(image, tof_tokens, return_aux=True)  # training only
```

The legacy three-input call remains accepted. At 640×480, the student has
approximately 0.353G MACs (convolution, linear, and attention). On the local
RTX 4070 Ti implementation check it used 68.5 MiB peak allocated memory and
4.34 ms p95; phone latency still needs measurement on the intended runtime and
device.

The implementation and experiment sequence are described in
[the teacher/student research plan](docs/teacher_student_architecture_plan.md).

![RGB–ToF teacher–student architecture](docs/depth_refinement_architecture.png)

## Legacy baseline (attention_v3)

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

Teacher training additionally requires the lazy, non-deployment dependency:

    python -m pip install -r requirements-teacher.txt

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

### Teacher/student training workflow

Training has two sequential stages. The teacher must finish first because its
best checkpoint supplies the online supervision used to train the student:

```text
configs/teacher_v4.yml
  → train teacher
  → checkpoints_teacher_v4/best.pt
  → load and freeze teacher while training student
  → checkpoints_student_v4/best.pt
```

#### Stage A: train the teacher

Install the optional teacher dependency and start training:

```bash
python -m pip install -r requirements-teacher.txt
python train.py --config configs/teacher_v4.yml
```

The teacher configuration activates the teacher in two places:

```yaml
model:
  architecture: teacher_v4
training:
  objective: teacher_v4
```

`architecture: teacher_v4` constructs `RGBToFTeacher`. The Depth Anything V2
Large backbone and neck are kept in evaluation mode, have gradients disabled,
and execute under `torch.no_grad()`. Its lightweight channel projections, the
ToF fusion modules, decoder, and metric-depth head remain trainable.

Stage A runs for 30 epochs with metric MSE plus a 0.1-weighted multi-scale
log-depth gradient loss. Corrected validation RMSE selects
`checkpoints_teacher_v4/best.pt`. TensorBoard events are written to
`runs/teacher_v4`:

```bash
tensorboard --logdir runs/teacher_v4
```

Resume an interrupted run with the same configuration and a saved epoch:

```bash
python train.py \
  --config configs/teacher_v4.yml \
  --resume checkpoints_teacher_v4/depth_refinement_epoch_010.pt \
  --epochs 30
```

Evaluate on validation data before beginning student training:

```bash
python eval.py \
  --config configs/teacher_v4.yml \
  --checkpoint checkpoints_teacher_v4/best.pt \
  --split val
```

#### Stage B: train the student

After the teacher checkpoint exists, start student distillation:

```bash
python train.py --config configs/student_v4.yml
```

The student configuration activates the deployable model and separately
describes the frozen teacher:

```yaml
model:
  architecture: student_v4
training:
  objective: student_distillation_v4
  distillation:
    teacher_checkpoint: checkpoints_teacher_v4/best.pt
    teacher_model:
      architecture: teacher_v4
```

The top-level model is the trainable student. At startup, `train.py` constructs
a separate teacher from `distillation.teacher_model`, strictly restores
`teacher_checkpoint`, freezes every teacher parameter, and keeps it in
evaluation mode. The teacher runs online under `torch.no_grad()` so image
augmentation and flipped ToF geometry remain aligned. Training-only 1×1
feature adapters are optimized with the student but are not registered inside
the deployable student model.

Student Stage B runs for 60 epochs:

- Epochs 1–5 use ground-truth losses only and do not execute the teacher.
- Epochs 6–9 linearly ramp teacher supervision.
- Epoch 10 onward uses the full teacher-depth and feature losses.

Pixels outside valid ToF footprints receive 2× initial supervision weight.
Teacher confidence is `exp(-|teacher-target| / 0.25)` and is applied only where
ground truth is valid. Checkpoints are written to `checkpoints_student_v4`, and
TensorBoard events are written to `runs/student_v4`:

```bash
tensorboard --logdir runs/student_v4
```

Resume student training with:

```bash
python train.py \
  --config configs/student_v4.yml \
  --resume checkpoints_student_v4/depth_refinement_epoch_020.pt \
  --epochs 60
```

The Depth Anything weights are external pretrained initialization; ZJU-L5 is
the only dataset used for task-specific teacher and student optimization.

## Evaluation

Evaluate the best checkpoint on the held-out test scenes:

    python eval.py --checkpoint checkpoints_attention_v3/best.pt

Evaluate the v4 teacher or student by supplying the matching configuration and
checkpoint. Use the validation split during development and reserve the test
split for final reporting:

```bash
python eval.py \
  --config configs/teacher_v4.yml \
  --checkpoint checkpoints_teacher_v4/best.pt \
  --split val

python eval.py \
  --config configs/student_v4.yml \
  --checkpoint checkpoints_student_v4/best.pt \
  --split test
```

Evaluation constructs only the architecture selected by the top-level
`model.architecture`. Student evaluation therefore loads only student weights;
it does not instantiate the teacher or the training-only feature adapters.

The evaluator reports pooled and image-averaged RMSE, MAE, AbsRel, and δ1,
plus the same pooled metrics for each depth bin and for pixels inside/outside
valid ToF coverage. Boundary accuracy is the fraction of valid target
discontinuity pairs above 0.1 m whose predicted discontinuity is also above
0.1 m. Values outside `[min_depth, max_depth]` are excluded. The old
clamp-to-range behavior is available only by setting
`legacy_clamp_out_of_range: true`, and those results must be labeled as legacy.
Save RGB, calibrated ToF, prediction, and ground-truth panels with:

    python eval.py --checkpoint checkpoints_attention_v3/best.pt --visualize

The legacy baseline remains in `config.yml`; v4 settings are in `configs/`.

## Student export

Export a fixed 480×640 TorchScript graph after student training. The exporter
strictly loads only student weights and checks PyTorch/TorchScript parity before
writing the artifact:

    python export_student.py \
      --config configs/student_v4.yml \
      --checkpoint checkpoints_student_v4/best.pt \
      --output student_v4_480x640.pt

The training-only teacher and feature adapters are not registered inside the
student and therefore cannot enter the exported graph.
