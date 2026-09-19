# Depth Refinement Model

This model takes an RGB image and a lower-resolution sparse depth map and
produces a dense depth map at the image resolution.

## Environment setup

A project-local `.venv` is already created and excluded from Git. Activate it:

```bash
cd /home/himwong/Desktop/depth_refine
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To recreate the environment from scratch:

```bash
cd /home/himwong/Desktop/depth_refine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that the model and dataset can be constructed before training:

```bash
python main.py
```

## Architecture

The model creates **two separate shared `QKᵀ` channel-attention matrices**.
`Aenc` comes from RGB and interpolated sparse depth and is shared by the encoder
levels. `Adec` uses the same two inputs but independent Q/K projection weights,
and is shared only by decoder concatenations. Every encoder level and decoder
concatenation has an independent value projection.

```mermaid
flowchart TB
    RGB["RGB image<br/>B × 3 × H × W"]
    SD["Sparse depth<br/>B × 1 × Hd × Wd"]
    RESIZE["Interpolate depth<br/>B × 1 × H × W"]
    SD --> RESIZE

    subgraph QK["Encoder shared QKᵀ — computed once"]
        direction LR
        QE["Image convolution<br/>and spatial pooling"]
        KE["Depth convolution<br/>and spatial pooling"]
        Q["Q<br/>B × Cattn × E"]
        K["K<br/>B × Cattn × E"]
        A["Encoder attention Aenc<br/>softmax(QKᵀ / √E)<br/>B × Cattn × Cattn"]

        QE --> Q
        KE --> K
        Q --> A
        K --> A
    end

    RGB --> QE
    RESIZE --> KE

    subgraph ENCODER["U-Net Encoder — independent V at each level"]
        direction TB

        E1["Level 1 DoubleConv<br/>X₁: B × C × H × W"]
        V1["Independent value projection<br/>V₁: B × Cattn × HW"]
        F1["F₁ = Norm(X₁ + Proj(Aenc · V₁))"]

        E2["Level 2 Down + DoubleConv<br/>X₂: B × 2C × H/2 × W/2"]
        V2["Independent value projection<br/>V₂: B × Cattn × H/2W/2"]
        F2["F₂ = Norm(X₂ + Proj(Aenc · V₂))"]

        E3["Level 3 Down + DoubleConv<br/>X₃: B × 4C × H/4 × W/4"]
        V3["Independent value projection<br/>V₃: B × Cattn × H/4W/4"]
        F3["F₃ = Norm(X₃ + Proj(Aenc · V₃))"]

        E4["Bottleneck Down + DoubleConv<br/>X₄: B × 8C × H/8 × W/8"]
        V4["Independent value projection<br/>V₄: B × Cattn × H/8W/8"]
        F4["F₄ = Norm(X₄ + Proj(Aenc · V₄))"]

        E1 --> V1 --> F1
        F1 --> E2 --> V2 --> F2
        F2 --> E3 --> V3 --> F3
        F3 --> E4 --> V4 --> F4
    end

    RGB --> E1

    A ==>|"same Aenc"| F1
    A ==>|"same Aenc"| F2
    A ==>|"same Aenc"| F3
    A ==>|"same Aenc"| F4

    subgraph DQK["Decoder shared QKᵀ — separate and computed once"]
        direction LR
        DQE["RGB image<br/>independent decoder query projection"]
        DKE["Interpolated depth<br/>independent decoder key projection"]
        DQ["Qdec<br/>B × Cattn × Edec"]
        DK["Kdec<br/>B × Cattn × Edec"]
        DA["Decoder attention Adec<br/>softmax(Qdec Kdecᵀ / √Edec)<br/>B × Cattn × Cattn"]

        DQE --> DQ
        DKE --> DK
        DQ --> DA
        DK --> DA
    end

    RGB --> DQE
    RESIZE --> DKE

    subgraph DECODER["U-Net Decoder — independent V for each skip concatenation"]
        direction TB
        C3["C₃ = Upsample(F₄) concat F₃<br/>B × 12C × H/4 × W/4"]
        DV3["Independent decoder value<br/>Vᵈ₃ = Proj(C₃)"]
        DF3["G₃ = Norm(C₃ + Proj(Adec · Vᵈ₃))"]
        U3["DoubleConv → D₃: 4C"]

        C2["C₂ = Upsample(D₃) concat F₂<br/>B × 6C × H/2 × W/2"]
        DV2["Independent decoder value<br/>Vᵈ₂ = Proj(C₂)"]
        DF2["G₂ = Norm(C₂ + Proj(Adec · Vᵈ₂))"]
        U2["DoubleConv → D₂: 2C"]

        C1["C₁ = Upsample(D₂) concat F₁<br/>B × 3C × H × W"]
        DV1["Independent decoder value<br/>Vᵈ₁ = Proj(C₁)"]
        DF1["G₁ = Norm(C₁ + Proj(Adec · Vᵈ₁))"]
        U1["DoubleConv → D₁: C"]
        HEAD["1 × 1 convolution<br/>Depth correction"]

        C3 --> DV3 --> DF3 --> U3
        U3 --> C2 --> DV2 --> DF2 --> U2
        U2 --> C1 --> DV1 --> DF1 --> U1 --> HEAD
    end

    F4 --> C3
    F3 -.->|"skip F₃"| C3
    F2 -.->|"skip F₂"| C2
    F1 -.->|"skip F₁"| C1

    DA ==>|"same Adec"| DF3
    DA ==>|"same Adec"| DF2
    DA ==>|"same Adec"| DF1

    ADD["Residual addition"]
    OUT["Dense depth<br/>B × 1 × H × W"]
    HEAD --> ADD
    RESIZE -->|"resized sparse depth"| ADD
    ADD --> OUT

    classDef image fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef depth fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef attention fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef value fill:#ffedd5,stroke:#ea580c,color:#7c2d12;
    classDef output fill:#f3e8ff,stroke:#9333ea,color:#581c87;

    class RGB,QE,Q,E1,E2,E3,E4,DQE,DQ image;
    class SD,RESIZE,KE,K,DKE,DK depth;
    class A,DA attention;
    class V1,V2,V3,V4,DV1,DV2,DV3 value;
    class F1,F2,F3,F4,C1,C2,C3,DF1,DF2,DF3,U3,U2,U1,HEAD,ADD,OUT output;
```

The attention operations are:

```text
Qenc = image embedding              # B × Cattn × E
Kenc = depth embedding              # B × Cattn × E
Aenc = softmax(Qenc Kencᵀ / √E)     # encoder matrix, computed once

V₁ = value_projection₁(X₁)          # independent weights
V₂ = value_projection₂(X₂)          # independent weights
V₃ = value_projection₃(X₃)          # independent weights
V₄ = value_projection₄(X₄)          # independent weights

Fᵢ = GroupNorm(Xᵢ + output_projectionᵢ(Aenc · Vᵢ))

Qdec = decoder_image_embedding(RGB)          # independent decoder Q weights
Kdec = decoder_depth_embedding(resized_depth)# independent decoder K weights
Adec = softmax(Qdec Kdecᵀ / √Edec)           # decoder matrix, computed once

Cᵢ = concat(upsample(Dᵢ₊₁), Fᵢ)    # decoder input + encoder skip
Vᵈᵢ = decoder_value_projectionᵢ(Cᵢ) # independent decoder weights
Gᵢ = GroupNorm(Cᵢ + output_projectionᵈᵢ(Adec · Vᵈᵢ))
Dᵢ = DoubleConvᵢ(Gᵢ)
```

The default encoder channel widths are `32 → 64 → 128 → 256`. Encoder and
decoder value projections both use `Cattn = 32`, but they consume their own
separate shared matrices (`Aenc` and `Adec`). Set `model.decoder_attention` to
`false` to disable decoder attention.

Checkpoints created before decoder attention was added do not contain the new
decoder Q/K/V weights. They remain usable with `decoder_attention: false`; train
from scratch with `decoder_attention: true` to use the new architecture.

## Configuration

Model and summary settings are controlled by [`config.yml`](config.yml). Run:

```bash
python main.py
```

To use another configuration:

```bash
python main.py --config path/to/config.yml
```

For the smaller iPhone-oriented variant, set both `base_channels` and
`attention_channels` to `16`. Set `summary.parameter_dtype` to `float16` to
report the expected FP16 weight size.

## ZJU-L5 DataLoader

The `data` section of [`config.yml`](config.yml) controls the ZJU-L5 input
pipeline. `main.py` builds the DataLoader after loading the configuration and
reports the first batch. Each batch contains:

- `image`: normalized RGB tensor, `[B, 3, 480, 640]`.
- `sparse_depth`: valid ToF mean depths, `[B, 1, 8, 8]`.
- `target_depth`: high-resolution target depth, `[B, 1, 480, 640]`.
- `sparse_valid_mask`: validity of the 64 ToF zones.
- `target_valid_mask`: validity of high-resolution depth pixels.
- `path`: source HDF5 path relative to the dataset root.

Inspect a batch with:

```bash
python main.py
```

## Training

Training is handled by `train.py`. `main.py` only validates the configuration,
constructs the model, and inspects one dataset batch.

Before starting, check these sections in [`config.yml`](config.yml):

```yaml
model:
  # Use 16 for a lighter model or 32 for the default model.
  base_channels: 16
  attention_channels: 16
  decoder_attention: true
  positive_output: true

data:
  root: /home/himwong/Downloads/ZJUL5/ZJUL5
  split: train
  batch_size: 1
  num_workers: 0

training:
  device: auto             # CUDA, MPS, or CPU is selected automatically
  epochs: 60
  learning_rate: 0.0003
  lr_scheduler: cosine
  warmup_epochs: 3
  warmup_start_factor: 0.1
  minimum_learning_rate: 0.000001
  loss: scale_invariant
  scale_invariant_lambda: 0.85
  scale_invariant_alpha: 10.0
  minimum_depth: 0.001
  amp: true                # used on CUDA only
  validation_split: test
  tensorboard:
    enabled: true
    log_dir: runs/depth_refinement
    image_every: 1
    num_images: 2
    depth_min: 0.0
    depth_max: null
  checkpoint_dir: checkpoints
  save_every: 5
```

Start training:

```bash
source .venv/bin/activate
python train.py
```

### TensorBoard

TensorBoard logging is enabled by default. Event files are written to
`runs/depth_refinement/` and contain:

- `Loss/train`: mean training loss for each epoch.
- `Loss/validation`: mean validation loss using the configured loss function.
- `Metrics/validation_MAE` and `Metrics/validation_RMSE` in meters.
- `Learning_rate/epoch`: the learning rate used during that epoch.
- `Samples/validation_*`: RGB, sparse depth, refined depth, and ground-truth
  panels generated with the same renderer as `eval.py`.

Start the TensorBoard web interface in another terminal while training runs:

```bash
source .venv/bin/activate
tensorboard --logdir runs/depth_refinement
```

Then open `http://localhost:6006`. `image_every` controls the sampling interval,
and `num_images` controls how many validation samples are written each time.
When `depth_max` is `null`, each panel uses the 99th percentile of its valid
ground truth as the upper color limit. Set `training.tensorboard.enabled` to
`false` to disable event logging. The `runs/` directory is ignored by Git.

The default loss follows DELTAR Equation 4: a scaled scale-invariant loss over
the logarithmic difference between predicted and target depth. It uses
`lambda = 0.85` and `alpha = 10`, ignores zero and non-finite target pixels,
and clamps valid depths to at least `0.001 m` before taking logarithms. Positive
model output is therefore enabled. At each configured save interval, a
checkpoint containing the model, optimizer, epoch, and training configuration
is written to `checkpoints/`. Validation reports the configured validation loss,
masked MAE, and masked RMSE.

The default learning-rate schedule linearly warms up from 10% to 100% of the
configured learning rate over three epochs, then uses cosine decay down to
`0.000001`. Set `lr_scheduler: none` to use a fixed learning rate.

The checkpoint with the lowest validation RMSE is also saved as:

```text
checkpoints/best.pt
```

Epoch checkpoints contain the model, optimizer, learning-rate scheduler, CUDA
AMP scaler, completed epoch, and best-validation state. Resume from one while
setting the total target epoch count with:

```bash
python train.py \
  --resume checkpoints/depth_refinement_epoch_010.pt \
  --epochs 30
```

This starts at epoch 11 and continues through epoch 30. You can also resume
from `checkpoints/best.pt`. Alternatively, set `training.resume_from` and
`training.epochs` in `config.yml`.

### Pipeline smoke test

Before a long run, temporarily use the following settings:

```yaml
training:
  device: cpu
  epochs: 1
  max_train_batches: 1
  max_validation_batches: 1
```

Then run:

```bash
python train.py
```

After the single train and validation batches complete successfully, restore
`device: auto`, the desired epoch count, and both `max_*_batches` values to
`null`. You can inspect the smoke-test metrics and sample inference panel in
TensorBoard as well.

Full `480 × 640` training is compute-intensive. Begin with
`base_channels: 16`, `attention_channels: 16`, and `batch_size: 1`. Increase
the batch size only after checking GPU memory usage.

## Evaluation

Evaluate a checkpoint produced by `train.py`:

```bash
source .venv/bin/activate
python eval.py --checkpoint checkpoints/depth_refinement_epoch_020.pt
```

Evaluation uses the `evaluation` section of [`config.yml`](config.yml) and
reports masked MAE and RMSE in meters. The checkpoint must use the same model
settings as the current `model` configuration.

Useful overrides:

```bash
# One-batch evaluation smoke test
python eval.py \
  --checkpoint checkpoints/depth_refinement_epoch_020.pt \
  --split test \
  --device cpu \
  --max-batches 1
```

Save side-by-side RGB, sparse ToF depth, refined depth, and ground-truth
visualizations during the same evaluation pass:

```bash
python eval.py \
  --checkpoint checkpoints/depth_refinement_epoch_020.pt \
  --visualize \
  --visualization-dir evaluation_outputs \
  --num-visualizations 8
```

All three depth panels use the same metric color scale. Invalid sparse and
ground-truth pixels are black, and each refined panel reports its per-image
masked MAE. Visualization defaults can also be set in the `evaluation` section
of [`config.yml`](config.yml).
