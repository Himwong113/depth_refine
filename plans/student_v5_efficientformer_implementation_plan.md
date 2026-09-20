# Student V5 EfficientFormer Implementation Plan

## Implementation status

The scratch EfficientFormerV2-S0 student, corrected v5 objective, supervised
control, configuration, TensorBoard component logging, deterministic seeding,
checkpoint metadata, TorchScript export, benchmark CLI, tests, and
documentation are implemented.

The untrained 480×640 graph contains 3,393,614 parameters. On the local RTX
4070 Ti it measured 7.22 ms p50, 7.86 ms p95, and 85.6 MiB peak allocated CUDA
memory. The fixed-resolution TorchScript smoke export had a maximum absolute
difference of 1.91e-6 from PyTorch.

Full 60-epoch ablations and Core ML measurements on the target iPhone remain
experiment and hardware gates.

## 1. Goal

Build a new deployable RGB–ToF depth student using a scratch-initialized
EfficientFormerV2-S0 RGB encoder, calibrated ToF cross-attention, and a
lightweight high-resolution decoder. Keep `student_v4` unchanged as the
reproducible baseline.

The new student may exceed the old 500,000-parameter limit. Deployment
acceptance is based on measured accuracy, sustained phone latency, and runtime
memory rather than parameter count alone.

### Current evidence

The existing best checkpoints were evaluated on the same 101-image validation
split:

| Metric | Student V4 | Teacher V4 |
| --- | ---: | ---: |
| Overall RMSE | 0.798 m | 0.550 m |
| Outside-ToF RMSE | 1.032 m | 0.719 m |
| Inside-ToF RMSE | 0.364 m | 0.227 m |
| Delta-1 | 66.1% | 87.9% |

Additional diagnostics:

- V4 RMSE was 0.422 m on a deterministic 101-image training subset but
  0.798 m on validation, indicating a substantial generalization gap.
- Disabling the trained ToF fusion modules increased validation RMSE from
  0.798 m to 0.962 m. The ToF attention is useful and must be retained.
- Uniformly reducing the current teacher-confidence weights does not reduce the
  normalized distillation loss or its total gradient because confidence appears
  in both the numerator and denominator.

## 2. Acceptance Criteria

Use the saved V4 checkpoint as the initial comparison:

| Requirement | V4 reference | V5 acceptance target |
| --- | ---: | ---: |
| Overall validation RMSE | 0.798 m | At most 0.758 m (5% improvement) |
| Outside-ToF validation RMSE | 1.032 m | At most 0.929 m (10% improvement) |
| Output resolution | 640×480 | 640×480 |
| Sustained throughput | Not phone-verified | At least 30 FPS |
| End-to-end model-path latency | Not phone-verified | p95 at most 33 ms |
| Incremental inference memory | Not phone-verified | At most 128 MiB |

Target model inference at 25 ms or less to leave time for preprocessing. Report
camera acquisition, the external ToF sensor update rate, and display latency
separately.

The target iPhone model is not yet specified. Training and desktop profiling
may proceed, but the model must not be described as real-time on iPhone until
it passes the sustained benchmark on the selected device.

## 3. Student V5 Architecture

### 3.1 Scratch semantic encoder

Use the `efficientformerv2_s0` architecture with random initialization and
its classification heads removed. Resize the normalized 480×640 RGB image to a
fixed 256×320 working image before the semantic encoder. Both working
dimensions are divisible by 32, which is required by the implementation's
strided attention residual path.

Pin the initial implementation to `timm==1.0.29`. Wrap the library model
behind a project-local encoder class so dependency-specific behavior does not
leak into the decoder or deployment interfaces.

The wrapper must:

- expose four NCHW pyramid features;
- remove the classifier and distillation heads;
- support strict offline construction when a complete V5 checkpoint is loaded;
- record that the encoder was initialized from scratch;
- reject undocumented missing or unexpected checkpoint keys;
- preserve the repository's ImageNet mean and standard deviation;
- support fixed rectangular input resolution.

### 3.2 Full-resolution detail branch

Add a shallow detail branch to preserve boundaries lost by the lower-resolution
semantic path:

1. 3×3 convolution, stride 2, RGB to 16 channels;
2. depthwise-separable block, stride 2, 16 to 32 channels;
3. depthwise-separable refinement block, 32 to 32 channels.

This produces a 32-channel feature at 120×160 for the decoder's 1/4 skip.

### 3.3 Feature-pyramid contract

For a 480×640 input, map the EfficientFormer features into the existing
decoder channel contract:

| Source | Expected working shape | V5 integration |
| --- | --- | --- |
| Detail branch | 32 × 120 × 160 | Decoder 1/4 skip |
| EfficientFormer stage 1 | 32 × 60 × 80 | Project to 64 channels; decoder 1/8 skip |
| EfficientFormer stage 2 | 48 × 30 × 40 | Project to 96 channels; decoder 1/16 skip |
| EfficientFormer stage 3 | 96 × 15 × 20 | Merge into deep representation |
| EfficientFormer stage 4 | 176 × 8 × 10 | Project, resize, and fuse into 128-channel 1/32 representation |

Keep the decoder-facing widths `[32, 64, 96, 128]`. This preserves the
current lightweight decoder layout and minimizes changes to feature
distillation.

Do not assume every library version produces these shapes. Add assertions with
clear errors and test the exact pinned implementation.

### 3.4 Fixed rectangular EfficientFormer attention

The upstream attention implementation contains resolution-dependent relative
position tables, indices, and strided upsampling. Construct the model for the
fixed 256×320 working resolution:

- initialize attention grids for 256×320 rather than a square input;
- regenerate non-persistent relative-position indices;
- clear cached attention biases after training/evaluation mode changes;
- require working-image dimensions divisible by 32;
- verify feature sizes 64×80, 32×40, 16×20, and 8×10.

The encoder constructor must always pass `pretrained=False`; it must reject a
configuration that requests pretrained weights.

### 3.5 ToF fusion

Keep the footprint-aware calibrated ToF cross-attention at decoder scales 1/16
and 1/8:

- preserve the 64×7 token interface;
- preserve invalid-token masking and the learned null token;
- preserve geometry-biased heads and metric ToF depth values;
- preserve the metric-depth anchor;
- initially retain `attention_dim=32`, two attention heads, and one geometry
  head.

The EfficientFormer attention and the ToF attention have different roles:
EfficientFormer models spatial RGB context, while ToF cross-attention injects
sensor measurements into image locations.

Use the RGB-only decoded 1/16 feature before sensor fusion as the
token-appearance source. Compute pooled token appearance once and pass the
result to both fusion layers, avoiding duplicated footprint-mask construction
and pooling.

### 3.6 Decoder and depth head

Retain additive skip decoding at 1/16, 1/8, and 1/4. Retain the narrow
full-resolution head initially:

- project the 1/4 decoder feature to eight channels;
- bilinearly resize it to 480×640;
- concatenate the original normalized RGB image;
- apply one depthwise-separable refinement block and one depth projection;
- add the ToF metric-depth anchor;
- enforce positive output with the existing softplus.

Do not widen the decoder during the first architecture experiment. This keeps
the effect of the EfficientFormerV2 architecture separable from decoder
capacity.

## 4. Public Interfaces and Configuration

Preserve the deployment APIs:

```python
depth = model(image, tof_tokens)
output = model(image, tof_tokens, return_aux=True)
```

`DepthOutput` continues to contain:

- full-resolution metric depth;
- 1/8 fused decoder feature;
- 1/16 fused decoder feature.

Add these identifiers:

```yaml
model:
  architecture: student_v5

training:
  objective: student_supervised_v5
# or
  objective: student_distillation_v5
```

The corrected supervised and distillation losses should also accept
`student_v4` so the weighting fix can be isolated from the encoder change.
The supervised objective must not construct a teacher or feature adapters.

Add V5 configuration fields:

```yaml
model:
  encoder_name: efficientformerv2_s0
  encoder_input_size: [256, 320]
  encoder_pretrained: false
  encoder_local_files_only: false
  encoder_channels: [32, 48, 96, 176]
  decoder_channels: [32, 64, 96, 128]
  detail_channels: 32
  refine_channels: 8

training:
  learning_rate: 0.0003
```

Validate all channel lists, image sizes, objective/architecture combinations,
and checkpoint architecture metadata before training begins.

## 5. Corrected Distillation

Keep `student_distillation_v4` available for checkpoint reproduction.
Implement a separately named corrected objective rather than silently changing
historical behavior.

### 5.1 Teacher-depth confidence

For valid pixels, define:

- `w=1` inside valid ToF footprints and `w=2` outside;
- `c=exp(-abs(teacher-target)/0.25)`;
- `e` as the student–teacher Smooth L1 error.

Use:

```text
teacher_depth_loss = sum(valid * w * c * e) / sum(valid * w)
```

Do not include confidence in the denominator. This makes low teacher confidence
reduce the absolute teacher contribution rather than only redistributing it.
Perform reductions in FP32 and return a finite zero for an empty valid mask.

### 5.2 Feature supervision

Replace nearest-neighbor validity reduction with area-pooled supervision:

```text
base_weight      = area_pool(valid * coverage_weight)
confident_weight = area_pool(valid * coverage_weight * teacher_confidence)

feature_loss = sum(confident_weight * cosine_error) / sum(base_weight)
```

Distill the fused 1/8 and 1/16 decoder features. Resize teacher features to the
student grid, detach all teacher tensors, and retain training-only 1×1 channel
adapters.

### 5.3 Initial loss weights

Keep the first experiment controlled:

| Component | Weight |
| --- | ---: |
| Ground-truth metric MSE | 1.0 |
| Ground-truth log-depth gradient | 0.1 |
| Teacher-depth Smooth L1 | 0.5 |
| Teacher-feature cosine | 0.05 |

Log each component separately, along with distillation strength and teacher
confidence statistics overall, inside ToF coverage, and outside coverage.

Do not label validation MSE minus the mixed training objective as an
overfitting gap. If a generalization metric is desired, evaluate the same
ground-truth metric on deterministic train and validation subsets.

## 6. Training Schedule

Reuse the existing frozen Teacher V4 best checkpoint for the first experiment
series.

| Epochs | Encoder state | Supervision |
| --- | --- | --- |
| 1–5 | Encoder trainable from random initialization | Ground truth only |
| 6–10 | Encoder trainable | Ramp depth and feature distillation from zero to full weight |
| 11–60 | Encoder trainable | Full configured supervision |

Optimization defaults:

- AdamW with weight decay `1e-4`;
- encoder, projections, detail branch, fusion, decoder, and adapters at `3e-4`;
- three warm-up epochs followed by cosine decay;
- CUDA AMP when available;
- gradient norm clipping at 1.0;
- batch size one initially.

Train the encoder from epoch one. EfficientFormer BatchNorm statistics and
affine parameters follow ordinary training behavior. Log their stability
because the initial batch size is one.

## 7. Reproducibility and Checkpoints

Seed Python, NumPy, Torch, CUDA, and the dataloader before model construction.

Every V5 checkpoint must save:

- architecture and complete resolved model configuration;
- complete resolved data and training configuration;
- encoder architecture and scratch initialization provenance;
- seed and objective version;
- model, optimizer, scheduler, and AMP scaler state;
- feature-adapter state when applicable;
- completed epoch, best metric, and best epoch;
- Python, NumPy, Torch, and CUDA RNG states.

Resume must restore the exact phase and optimizer groups. Evaluation and export
must build directly from checkpoint metadata or validate the supplied config
against it. A complete checkpoint must load without downloading model weights.

Use new directories such as:

```text
checkpoints_student_v5_supervised/
checkpoints_student_v5_distilled/
runs/student_v5_supervised/
runs/student_v5_distilled/
```

Require a real validation split for research training. Do not silently use the
test split for model selection.

## 8. Controlled Experiment Matrix

Use identical splits, augmentation, schedules, and evaluation code.

| Run | Architecture | Objective | Question |
| --- | --- | --- | --- |
| A | V4 | Original distillation | Historical reference |
| B | V4 | Corrected distillation | Does corrected confidence weighting help? |
| C | Scratch V5 | Ground truth only | Does the new RGB architecture help without the teacher? |
| D | Scratch V5 | Ground truth plus depth KD | Does prediction transfer help? |
| E | Scratch V5 | Ground truth plus depth and feature KD | Does feature transfer add value? |

Run the first sweep with seed 42. Repeat B, C, and the strongest distilled V5
variant using seeds 42, 43, and 44. Report the mean and spread, not only the
best checkpoint.

Select checkpoints using validation overall RMSE. Before promotion, also check
outside-coverage and boundary metrics. Do not inspect test results while
choosing architecture or hyperparameters.

For each run report:

- RMSE, MAE, AbsRel, and delta-1 overall;
- the same metrics inside and outside ToF coverage;
- depth-bin metrics for 0–2 m, 2–4 m, 4–6 m, and over 6 m;
- depth-boundary RMSE, delta-1, and boundary recall;
- parameters, MACs, peak memory, and desktop latency.

Produce matched validation figures using the same images and color limits:
RGB, ToF coverage, target, teacher, V4, V5, and absolute-error maps.

## 9. Export and Real-Time Validation

### 9.1 Early export gate

Before full training:

1. Construct the randomly initialized V5 model.
2. Run a 480×640 forward and backward smoke test.
3. Export a fixed-shape TorchScript model.
4. Convert the complete graph to Core ML.
5. Verify output shape and numerical parity.
6. Inspect the Core ML compute plan for unsupported operations and device
   transfers.
7. If a target phone is available, measure an untrained model's runtime to
   validate architectural feasibility.

Do not spend a full training run on a graph that cannot be exported or misses
the latency budget structurally.

### 9.2 Export contract

The deployment model accepts:

```text
image:      float tensor [1, 3, 480, 640]
tof_tokens: float tensor [1, 64, 7]
depth:      float tensor [1, 1, 480, 640]
```

The exported artifact must contain neither the teacher nor training-only
feature adapters.

### 9.3 On-device benchmark

Benchmark batch size one using:

- 50 untimed warm-up predictions;
- at least 1,000 timed predictions;
- a five-minute sustained run for thermal behavior;
- the complete preprocessing, model, and postprocessing path;
- the same compute-unit policy intended for production.

Record the iPhone model, SoC, iOS version, Core ML version, model precision,
compute-unit policy, p50/p95 latency, throughput, and peak incremental memory.
Report cold startup separately.

### 9.4 Optimization order

If latency exceeds the target:

1. remove duplicated appearance pooling and tensor conversions;
2. fix unsupported operators or unnecessary CPU/GPU/Neural Engine transfers;
3. benchmark a 192×256 semantic encoder input while retaining 480×640 output;
4. retrain that lower-resolution candidate and repeat the accuracy comparison;
5. evaluate weight compression or quantization only after establishing the
   FP16 reference.

Do not move to EfficientFormerV2-S1 until S0 demonstrates accuracy improvement
and measured latency headroom.

## 10. Tests

### Architecture

- V5 returns the expected depth and auxiliary feature shapes for 480×640 input.
- Every rectangular attention layer uses the expected resolution.
- Construction performs no model-weight download.
- The full-resolution detail branch receives gradients.
- Frozen encoder weights receive no gradients during epochs 1–5.
- Encoder weights receive finite gradients after unfreezing.
- EfficientFormer BatchNorm statistics update during training.

### ToF behavior

- Invalid token contents cannot alter the prediction.
- An all-invalid token frame remains finite.
- Horizontal flipping transforms calibrated token geometry consistently.
- The shared appearance-pooling path matches the reference implementation.
- The trained sensor fusion can be enabled/disabled for diagnostic evaluation.

### Loss behavior

- Scaling all confidences from 1.0 to 0.1 scales the teacher contribution and
  its gradients by 0.1 when other terms are fixed.
- Empty valid masks return a finite zero.
- Invalid target pixels do not affect any distillation term.
- Area-pooled feature weights preserve partial valid coverage.
- Teacher tensors and weights never receive gradients.

### Checkpoint and resume

- V3 and V4 loading remains unchanged.
- V5 architecture mismatches fail clearly.
- Resume restores the optimizer, scheduler, scaler, adapters, and RNG states.
- Complete V5 evaluation works offline.

### Export

- Fixed-resolution TorchScript output matches PyTorch within the existing
  FP32 maximum tolerance of `1e-4`.
- Initial Core ML FP16 targets are mean absolute depth difference at most
  1 mm, maximum difference at most 2 cm, and validation RMSE degradation at
  most 0.5%.
- Exported models contain no teacher or adapter parameters.

Tolerance failures must be investigated rather than hidden by increasing the
threshold.

## 11. Implementation Order

1. Add corrected, versioned losses and their focused tests.
2. Add the pinned dependency and local EfficientFormer encoder wrapper.
3. Implement rectangular attention-bias conversion and shape tests.
4. Implement `student_v5`, its detail branch, projections, and shared
   appearance pooling.
5. Add V5 model factory, configurations, objective selection, deterministic
   seeding, and checkpoint metadata.
6. Extend evaluation and visualization for matched comparisons.
7. Complete the early TorchScript/Core ML export gate.
8. Run short overfit and smoke-training checks.
9. Run experiments A–E and repeated-seed validation.
10. Export the strongest candidate and complete sustained phone profiling.
11. Update the README, architecture documentation, architecture figure, and
    measured result tables.

## 12. Promotion Rule

Promote V5 only if it:

- meets both validation improvement thresholds;
- does not regress invalid/all-invalid ToF behavior;
- passes export parity;
- sustains p95 latency at or below 33 ms on the named target phone;
- stays within 128 MiB incremental runtime memory.

Until phone measurements pass, describe the result as an accuracy candidate,
not a real-time iPhone deployment model.
