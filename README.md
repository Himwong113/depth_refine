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

The RGB and depth inputs create **one shared `QKᵀ` channel-attention matrix**.
Every U-Net encoder level has its own independent value projection `Vᵢ`, and
the shared attention matrix is applied to all of them.

```mermaid
flowchart TB
    RGB["RGB image<br/>B × 3 × H × W"]
    SD["Sparse depth<br/>B × 1 × Hd × Wd"]
    RESIZE["Interpolate depth<br/>B × 1 × H × W"]
    SD --> RESIZE

    subgraph QK["Shared QKᵀ Attention — computed once"]
        direction LR
        QE["Image convolution<br/>and spatial pooling"]
        KE["Depth convolution<br/>and spatial pooling"]
        Q["Q<br/>B × Cattn × E"]
        K["K<br/>B × Cattn × E"]
        A["Shared attention A<br/>softmax(QKᵀ / √E)<br/>B × Cattn × Cattn"]

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
        F1["F₁ = Norm(X₁ + Proj(A · V₁))"]

        E2["Level 2 Down + DoubleConv<br/>X₂: B × 2C × H/2 × W/2"]
        V2["Independent value projection<br/>V₂: B × Cattn × H/2W/2"]
        F2["F₂ = Norm(X₂ + Proj(A · V₂))"]

        E3["Level 3 Down + DoubleConv<br/>X₃: B × 4C × H/4 × W/4"]
        V3["Independent value projection<br/>V₃: B × Cattn × H/4W/4"]
        F3["F₃ = Norm(X₃ + Proj(A · V₃))"]

        E4["Bottleneck Down + DoubleConv<br/>X₄: B × 8C × H/8 × W/8"]
        V4["Independent value projection<br/>V₄: B × Cattn × H/8W/8"]
        F4["F₄ = Norm(X₄ + Proj(A · V₄))"]

        E1 --> V1 --> F1
        F1 --> E2 --> V2 --> F2
        F2 --> E3 --> V3 --> F3
        F3 --> E4 --> V4 --> F4
    end

    RGB --> E1

    A ==>|"same A"| F1
    A ==>|"same A"| F2
    A ==>|"same A"| F3
    A ==>|"same A"| F4

    subgraph DECODER["U-Net Decoder"]
        direction TB
        U3["Upsample + concatenate F₃<br/>DoubleConv → 4C"]
        U2["Upsample + concatenate F₂<br/>DoubleConv → 2C"]
        U1["Upsample + concatenate F₁<br/>DoubleConv → C"]
        HEAD["1 × 1 convolution<br/>Depth correction"]

        U3 --> U2 --> U1 --> HEAD
    end

    F4 --> U3
    F3 -.->|"skip F₃"| U3
    F2 -.->|"skip F₂"| U2
    F1 -.->|"skip F₁"| U1

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

    class RGB,QE,Q,E1,E2,E3,E4 image;
    class SD,RESIZE,KE,K depth;
    class A attention;
    class V1,V2,V3,V4 value;
    class F1,F2,F3,F4,U3,U2,U1,HEAD,ADD,OUT output;
```

The attention operations are:

```text
Q = image embedding                 # B × Cattn × E
K = depth embedding                 # B × Cattn × E
A = softmax(QKᵀ / √E)              # B × Cattn × Cattn, computed once

V₁ = value_projection₁(X₁)          # independent weights
V₂ = value_projection₂(X₂)          # independent weights
V₃ = value_projection₃(X₃)          # independent weights
V₄ = value_projection₄(X₄)          # independent weights

Fᵢ = GroupNorm(Xᵢ + output_projectionᵢ(A · Vᵢ))
```

The default encoder channel widths are `32 → 64 → 128 → 256`, while all value
projections map into the common `Cattn = 32` channel space so they can use the
same shared attention matrix.

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
  positive_output: true

data:
  root: /home/himwong/Downloads/ZJUL5/ZJUL5
  split: train
  batch_size: 1
  num_workers: 0

training:
  device: auto             # CUDA, MPS, or CPU is selected automatically
  epochs: 20
  learning_rate: 0.0003
  loss: scale_invariant
  scale_invariant_lambda: 0.85
  scale_invariant_alpha: 10.0
  minimum_depth: 0.001
  amp: true                # used on CUDA only
  validation_split: test
  checkpoint_dir: checkpoints
  save_every: 5
```

Start training:

```bash
source .venv/bin/activate
python train.py
```

The default loss follows DELTAR Equation 4: a scaled scale-invariant loss over
the logarithmic difference between predicted and target depth. It uses
`lambda = 0.85` and `alpha = 10`, ignores zero and non-finite target pixels,
and clamps valid depths to at least `0.001 m` before taking logarithms. Positive
model output is therefore enabled. At each configured save interval, a
checkpoint containing the model, optimizer, epoch, and training configuration
is written to `checkpoints/`. Validation reports masked MAE and RMSE.

The checkpoint with the lowest validation RMSE is also saved as:

```text
checkpoints/best.pt
```

Epoch checkpoints contain the model, optimizer, CUDA AMP scaler, completed
epoch, and best-validation state. Resume from one while setting the total target
epoch count with:

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
`null`.

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
# depth_refine
