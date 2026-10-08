# FedEVT-OSAD
## Federated Extreme-Value Open-Set Anomaly Detection

**BiLSTM + Temporal Attention + Autoencoder Backbone for Zero-Day Cyber Threat Detection**

> A research-oriented anomaly detection backbone designed to learn temporal representations of network traffic without relying on attack-family labels, enabling evaluation against previously unseen attack families.

---

## Overview

**FedEVT-OSAD** is a research framework for **open-set and zero-day anomaly detection in network security environments**.

The current implementation establishes the core neural representation-learning backbone:

```text
Network Traffic
      │
      ▼
Data Cleaning & Leakage Control
      │
      ▼
Feature Engineering
      │
      ▼
Train-only Scaling + One-Hot Encoding
      │
      ▼
Temporal Windows
      │
      ▼
BiLSTM Encoder
      │
      ▼
Temporal Attention
      │
      ▼
64-Dimensional Latent Representation
      │
      ▼
LSTM Decoder
      │
      ▼
Reconstructed Traffic Sequence
      │
      ▼
Reconstruction Error
      │
      └──────────────► Future EVT Novelty Score
```

The design is intentionally structured around **generalization rather than closed-set classification**.

Instead of asking:

> "Which known attack class does this sample belong to?"

the backbone asks:

> "Does this traffic sequence conform to the representation learned from previously observed traffic?"

This distinction is critical for detecting **previously unseen attack families**, where conventional supervised classifiers can fail because the target class was never available during training.

---

# Key Research Contributions

### 1. Zero-Day Evaluation Through LOFO

The experiment uses a **Leave-One-Family-Out (LOFO)** protocol.

For the current experiment:

```text
Held-out Family = WannaCry
```

The WannaCry family is completely excluded from the training/validation partition and evaluated as a **zero-day / unseen family**.

```text
                    Dataset
                       │
          ┌────────────┴────────────┐
          │                         │
    Known Families            WannaCry
          │                    Held Out
          │                         │
       Train/Val                Zero-Day
```

This provides a substantially more meaningful evaluation setting than randomly splitting individual samples from the same attack family across training and testing.

---

### 2. Leakage-Aware Representation Learning

Cybersecurity datasets frequently contain fields that can allow a model to memorize attack identities rather than learn behavioral patterns.

The pipeline therefore removes target-identifying information from the final model input.

In particular:

- `Family` is retained only for evaluation.
- `Family` is explicitly removed from model features.
- Network identifiers such as IP-related information are not used as direct predictive signals.
- Scaling is fitted **only on training data**.
- One-hot encoding is fitted **only on training data**.

This prevents information from the validation or held-out zero-day partition from influencing preprocessing.

The resulting model input contains **18 features**.

---

# Dataset

The notebook loads the dataset from:

```text
archive-2/final(2).csv
```

Initial dataset size:

```text
149,043 rows
14 raw columns
```

The dataset contains network/security attributes including:

- Time
- Protocol
- Flag
- Family
- Clusters
- BTC
- USD
- Netflow bytes
- Threat information
- Network identifiers

The dataset audit reported:

```text
Missing values: 0
Duplicate rows: 0
Ransomware families: 17
Prediction classes: 3
Threat categories: 9
```

---

# Data Processing Pipeline

## 1. Data Audit

The pipeline first inspects:

- Dataset dimensions
- Column types
- Missing values
- Duplicate records
- Numerical attributes
- Categorical attributes
- Attack-family cardinality
- Threat categories

This establishes a reproducible data-quality baseline before model training.

---

## 2. Feature Engineering

The model uses behavioral and temporal signals rather than attack-family identifiers.

### Numerical Features

The final numerical feature set includes:

```text
Clusters
BTC
USD
Netflow_Bytes
hour_sin
hour_cos
```

### Categorical Features

Categorical network attributes include:

```text
Protocol
Flag
```

These are transformed using one-hot encoding.

The final feature representation contains:

```text
18 dimensions
```

### Cyclical Time Encoding

Time-of-day information is represented using:

```text
hour_sin
hour_cos
```

This allows the model to represent temporal periodicity without creating an artificial discontinuity between adjacent times such as:

```text
23:00 → 00:00
```

---

# Train / Validation / Zero-Day Split

The experiment holds out the WannaCry attack family.

```text
Train / Validation
------------------
132,933 records

Zero-Day Test
-------------
16,110 records

Held-out Family
---------------
WannaCry
```

Importantly, the held-out family is removed **before temporal window construction**.

This ensures that the zero-day evaluation partition remains isolated from model fitting.

---

# Temporal Modeling

Network attacks are rarely defined by a single isolated observation.

Instead, malicious behavior can emerge through a sequence of events.

To capture this behavior, the model converts individual observations into sequential windows.

Configuration:

```text
Window Size = 10
```

Each training example therefore has the shape:

```text
10 timesteps × 18 features
```

Resulting temporal datasets:

| Partition | Windows | Shape |
|---|---:|---|
| Training | 106,337 | `(106337, 10, 18)` |
| Validation | 26,578 | `(26578, 10, 18)` |
| Zero-Day Test | 16,101 | `(16101, 10, 18)` |

The label associated with each temporal window corresponds to the label of its final timestep and is used for **post-hoc evaluation rather than supervised autoencoder training**.

---

# Model Architecture

The core model is a:

> **Bidirectional LSTM + Temporal Attention + Autoencoder**

The architecture contains three major components:

```text
             Input Sequence
             (10 × 18)
                  │
                  ▼
        ┌──────────────────┐
        │   BiLSTM Encoder │
        │   2 Layers       │
        │   Hidden = 64    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Temporal         │
        │ Attention        │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Latent Projection│
        │     z ∈ R⁶⁴      │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ LSTM Decoder     │
        │   2 Layers       │
        └────────┬─────────┘
                 │
                 ▼
        Reconstructed Sequence
             (10 × 18)
```

---

# BiLSTM Encoder

The encoder uses a bidirectional LSTM:

```text
Input Dimension = 18
Hidden Size     = 64
Layers          = 2
Bidirectional   = Yes
Dropout         = 0.2
```

Because the network is bidirectional, each timestep receives contextual information from both temporal directions.

The resulting hidden representation is:

```text
64 × 2 = 128 dimensions
```

which is subsequently processed by the attention mechanism.

---

# Temporal Attention

The attention module learns which timesteps contribute most strongly to the representation.

The attention mechanism is implemented as:

```text
Linear
   ↓
Tanh
   ↓
Linear
   ↓
Softmax over time
```

Mathematically:

\[
\alpha_t =
\frac{\exp(e_t)}
{\sum_j \exp(e_j)}
\]

where:

- \(e_t\) is the learned attention score for timestep \(t\)
- \(\alpha_t\) is the normalized temporal importance

The context vector is:

\[
c = \sum_t \alpha_t h_t
\]

where \(h_t\) is the BiLSTM representation at timestep \(t\).

This produces a sequence-level representation that can emphasize the most informative parts of a network-traffic window.

---

# Latent Representation

The attention context vector is projected into a compact latent space:

```text
Latent Dimension = 64
```

The encoder therefore produces:

\[
z = Encoder(x)
\]

where:

\[
z \in \mathbb{R}^{64}
\]

This latent representation serves as the stable interface for the future novelty-detection modules.

---

# Decoder

The decoder reconstructs the original temporal sequence from the latent representation.

Pipeline:

```text
64-D Latent Vector
        │
        ▼
Hidden Projection
        │
        ▼
Repeated Across 10 Timesteps
        │
        ▼
2-Layer LSTM
        │
        ▼
Output Projection
        │
        ▼
10 × 18 Reconstruction
```

The reconstruction objective forces the encoder to preserve information necessary to reproduce the learned traffic representation.

---

# Anomaly Scoring

The current backbone uses **reconstruction error** as its anomaly signal.

For an input sequence \(x\) and reconstruction \(\hat{x}\):

\[
R(x) =
\frac{1}{T F}
\sum_{t=1}^{T}
\sum_{f=1}^{F}
(x_{t,f}-\hat{x}_{t,f})^2
\]

where:

- \(T = 10\) timesteps
- \(F = 18\) features

Higher reconstruction error indicates that the observed sequence is harder for the learned representation to reconstruct.

The implementation computes a per-window MSE score.

---

# Training Strategy

The autoencoder is trained using reconstruction loss:

```text
Loss = Mean Squared Error
```

Optimizer:

```text
Adam
```

Configuration:

```text
Learning Rate = 1e-3
Weight Decay   = 1e-5
Batch Size     = 128
Epochs         = 50
Gradient Clip  = 1.0
```

A `ReduceLROnPlateau` scheduler is used:

```text
Factor   = 0.5
Patience = 5
```

The best model checkpoint is selected using validation reconstruction loss.

Checkpoint:

```text
results/checkpoints/best_model.pt
```

---

# Reproducibility

The experiment fixes the random seed:

```text
Seed = 42
```

The seed is applied to:

- Python random
- NumPy
- PyTorch
- CUDA when available

The experiment also persists its configuration:

```text
results/config.json
```

This makes the experiment easier to reproduce and compare across future runs.

---

# Training Results

The current run was performed on:

```text
Device: CPU
PyTorch: 2.14.0
CUDA: Not available
```

Training reconstruction loss decreased substantially during optimization.

Representative checkpoints:

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.106998 | 1.235973 |
| 5 | 0.033396 | 1.194202 |
| 10 | 0.021262 | 1.202231 |
| 20 | 0.013379 | 1.165965 |
| 30 | 0.009997 | 1.180100 |
| 40 | 0.009501 | 1.180259 |
| 50 | 0.009391 | 1.182657 |

Best recorded validation loss:

```text
1.160520
```

---

# Zero-Day Reconstruction Analysis

The current experiment evaluates the held-out WannaCry family against known-family validation traffic.

Observed median reconstruction errors:

```text
Known validation traffic:
0.009996

Held-out WannaCry:
0.007672
```

The current result **does not show elevated reconstruction error for WannaCry**.

This is an important research finding rather than something to hide in the README.

It indicates that reconstruction error alone is not yet a sufficiently discriminative zero-day signal for this experimental configuration.

The notebook therefore explicitly identifies the need for additional novelty signals and EVT calibration.

---

# Why This Result Matters

A strong research system should not only report successful experiments.

It should expose where the current hypothesis breaks.

The current experiment demonstrates that:

> A sophisticated temporal autoencoder can learn compact network representations, but reconstruction error by itself may not reliably separate an unseen attack family from known traffic.

This motivates the next stage of FedEVT-OSAD:

```text
Reconstruction Error
        +
Latent Prototype Distance
        +
Extreme Value Theory
        +
Federated Evidence Aggregation
```

The objective is to move beyond a single reconstruction metric toward a statistically calibrated open-set novelty score.

---

# Latent-Space Analysis

The model exposes the latent representation:

\[
z = Encoder(x)
\]

The notebook applies PCA to the learned latent vectors for visualization.

This enables investigation of:

- Known vs. unseen traffic separation
- Family-level clustering
- Latent representation compactness
- Potential overlap between attack families
- Whether unseen attacks occupy novel regions of representation space

Generated visualization:

```text
results/figures/latent_pca.png
```

---

# Attention Interpretability

The model also exposes temporal attention weights.

This enables inspection of:

```text
Which timesteps influenced the representation most?
```

Attention visualizations are generated for both known and zero-day samples.

Output:

```text
results/figures/attention_weights.png
```

This provides a lightweight interpretability mechanism for understanding temporal decision context.

---

# Baseline Model

To evaluate whether temporal attention provides meaningful representational value, the notebook also defines a simpler baseline:

> **LSTM Autoencoder**

The baseline uses:

- Unidirectional LSTM
- Same general latent/decoder structure
- No temporal attention

This creates a controlled comparison between:

```text
Baseline:
LSTM → Latent → Decoder

Proposed Backbone:
BiLSTM → Attention → Latent → Decoder
```

The comparison is designed to isolate the contribution of:

- Bidirectional temporal context
- Learned timestep weighting
- Attention-based sequence summarization

---

# Experiment Outputs

The notebook organizes generated artifacts under:

```text
results/
├── config.json
│
├── checkpoints/
│   └── best_model.pt
│
├── figures/
│   ├── training_curves.png
│   ├── reconstruction_boxplot.png
│   ├── reconstruction_hist.png
│   ├── latent_pca.png
│   └── attention_weights.png
│
├── metrics/
│
└── logs/
```

This separation is intentional so that experiment configuration, model checkpoints, visualizations, metrics, and logs can evolve independently.

---

# Configuration

The main research configuration is centralized in one dictionary:

```python
CONFIG = {
    "seed": 42,
    "window_size": 10,
    "hidden_size": 64,
    "num_layers": 2,
    "latent_dim": 64,
    "dropout": 0.2,
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "benign_label": None,
    "held_out_family": "WannaCry"
}
```

This makes experiments configurable without modifying the underlying architecture.

---

# Technology Stack

### Machine Learning

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn

### Deep Learning

- LSTM
- BiLSTM
- Temporal Attention
- Autoencoders
- PCA

### Data Processing

- StandardScaler
- OneHotEncoder
- Sliding-window temporal transformation

### Visualization

- Matplotlib
- Seaborn

---

# Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
FedEVT_OSAD_BiLSTM_Attention_AE.ipynb
```

---

# Running the Experiment

## 1. Prepare Dataset

Place the dataset at:

```text
archive-2/final(2).csv
```

or update:

```python
CONFIG["data_path"]
```

---

## 2. Select the Zero-Day Family

Change:

```python
CONFIG["held_out_family"] = "WannaCry"
```

For another LOFO experiment:

```python
CONFIG["held_out_family"] = "<ATTACK_FAMILY>"
```

The selected family will be removed from the training/validation partition.

---

## 3. Execute the Notebook

Run the notebook sequentially.

The pipeline will:

1. Audit the dataset.
2. Clean the input.
3. Engineer temporal features.
4. Split the held-out family.
5. Fit preprocessing on training data.
6. Construct temporal windows.
7. Train the BiLSTM-Attention-AE.
8. Save the best checkpoint.
9. Calculate reconstruction errors.
10. Analyze latent representations.
11. Visualize attention.
12. Compare against the LSTM-AE baseline.

---

# Research Design

The core research hypothesis can be summarized as:

\[
H_1:
\text{Unseen attack families produce representations that differ from known traffic.}
\]

The current implementation tests one part of this hypothesis:

\[
H_1' :
R(x)_{\text{zero-day}}
>
R(x)_{\text{known}}
\]

For the current WannaCry experiment, this condition was **not observed**.

Therefore, the next research question becomes:

\[
\text{Can latent-space distance + EVT calibration improve open-set detection?}
\]

---

# Next Phase: FedEVT-OSAD

The current notebook intentionally establishes a stable encoder/decoder interface.

The next version extends:

```text
z = Encoder(x)

R(x) = Reconstruction Error
```

into a multi-signal novelty framework.

---

## 1. Prototype Distance

Learn prototypes in latent space:

\[
D(x)=\min_k d(z,P_k)
\]

where:

- \(z\) = sample latent representation
- \(P_k\) = learned prototype
- \(d\) = latent-space distance

This provides information that reconstruction error alone cannot capture.

---

## 2. Fused Novelty Score

Combine reconstruction error and latent-space novelty:

\[
S(x)
=
\lambda R_{\text{normalized}}(x)
+
(1-\lambda)D_{\text{normalized}}(x)
\]

This allows the detector to consider both:

```text
Reconstruction abnormality
+
Representation-space novelty
```

---

## 3. Extreme Value Theory Calibration

Instead of relying on an arbitrary fixed anomaly threshold, the next phase will model the tail of the anomaly-score distribution.

Planned pipeline:

```text
Novelty Scores
      │
      ▼
95th Percentile Threshold u
      │
      ▼
Exceedances
      │
      ▼
Generalized Pareto Distribution
      │
      ▼
EVT Threshold τ
      │
      ▼
Calibrated Open-Set Decision
```

This is intended to provide statistically grounded thresholds for rare-event detection.

---

# Federated Extension

The ultimate architecture extends the local detector into a federated setting.

Conceptually:

```text
                ┌─────────────────────┐
                │    Global Server    │
                │                     │
                │ Federated Tail      │
                │ Aggregation         │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          Client A     Client B     Client C
              │            │            │
        Local Traffic Local Traffic Local Traffic
              │            │            │
              ▼            ▼            ▼
        Local Encoder  Local Encoder  Local Encoder
              │            │            │
              ▼            ▼            ▼
        Local Novelty  Local Novelty  Local Novelty
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                  Tail / Evidence Sharing
```

The planned aggregation mechanism includes:

- Federated tail sharing
- Evidence-weighted shrinkage
- Local anomaly distributions
- Cross-client novelty calibration

The goal is to improve generalization to attacks that may be observed by one organization but not another **without requiring raw traffic to be centrally pooled**.

---

# Engineering Principles

The project follows several principles important for production-grade ML systems.

### Reproducibility

Centralized configuration and deterministic seeding make experiments repeatable.

### Leakage Prevention

Preprocessing is fitted on training data only.

### Modular Architecture

Encoder, attention, decoder, scoring, evaluation, and future EVT modules are separable.

### Explicit Evaluation Protocol

LOFO provides a clear zero-day evaluation methodology.

### Observability

Training curves, reconstruction distributions, latent PCA, and attention weights provide visibility into model behavior.

### Honest Failure Analysis

The current experiment does not artificially claim successful zero-day detection when reconstruction error does not support that conclusion.

---

# Limitations

The current implementation should be interpreted as a **research backbone**, not a production intrusion-detection system.

Current limitations include:

1. EVT calibration is not yet implemented.
2. Federated aggregation is not yet implemented.
3. Prototype-distance novelty is not yet implemented.
4. Reconstruction error alone did not distinguish WannaCry as anomalous in the current experiment.
5. The current configuration trains the autoencoder on all non-held-out families unless `benign_label` is explicitly configured.
6. The current experiment runs on CPU.
7. Additional cross-family and repeated LOFO experiments are required before drawing broad generalization conclusions.
8. The current notebook is an experimental research implementation rather than a hardened production service.

These limitations define the next engineering and research milestones rather than invalidating the backbone.

---

# Project Structure

A recommended repository structure is:

```text
FedEVT-OSAD/
│
├── notebooks/
│   └── FedEVT_OSAD_BiLSTM_Attention_AE.ipynb
│
├── data/
│   └── README.md
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── attention/
│   ├── scoring/
│   ├── evt/
│   └── federated/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── checkpoints/
│   └── logs/
│
├── configs/
│
├── requirements.txt
│
└── README.md
```

---

# Reproducibility Checklist

Before reporting an experiment, record:

```text
[ ] Dataset version
[ ] Held-out attack family
[ ] Random seed
[ ] Feature configuration
[ ] Window size
[ ] Model configuration
[ ] Optimizer
[ ] Learning rate
[ ] Number of epochs
[ ] Best validation loss
[ ] Reconstruction-error distribution
[ ] Latent-space visualization
[ ] Baseline comparison
```

This makes experiments easier to reproduce and scientifically compare.

---

# Research Roadmap

### Phase 1 — Neural Backbone
- [x] Dataset audit
- [x] Leakage-aware preprocessing
- [x] LOFO evaluation
- [x] Temporal feature engineering
- [x] BiLSTM encoder
- [x] Temporal attention
- [x] Autoencoder decoder
- [x] Reconstruction scoring
- [x] Latent-space analysis
- [x] Attention visualization
- [x] LSTM-AE baseline

### Phase 2 — Open-Set Scoring
- [ ] Latent prototypes
- [ ] Prototype-distance scoring
- [ ] Reconstruction + distance fusion
- [ ] Cross-family evaluation

### Phase 3 — EVT
- [ ] Tail threshold selection
- [ ] Generalized Pareto fitting
- [ ] EVT-calibrated thresholds
- [ ] False-positive analysis
- [ ] Rare-event detection evaluation

### Phase 4 — Federated Learning
- [ ] Local detector training
- [ ] Federated tail statistics
- [ ] Evidence-weighted aggregation
- [ ] Client heterogeneity experiments
- [ ] Privacy-aware aggregation

### Phase 5 — Productionization
- [ ] Experiment configuration system
- [ ] Model registry
- [ ] Offline evaluation pipeline
- [ ] Streaming inference
- [ ] Monitoring
- [ ] Drift detection
- [ ] Model/version tracking
- [ ] Containerized deployment

---

# Why FedEVT-OSAD?

Traditional intrusion detection systems often assume that the attack classes encountered during deployment are represented during training.

Real-world security systems do not have that luxury.

New ransomware variants, previously unseen attack families, infrastructure changes, and adversarial behavior can invalidate a closed-set classifier.

FedEVT-OSAD approaches the problem differently:

```text
Learn normal/known behavioral structure
              ↓
Compress temporal behavior
              ↓
Measure reconstruction and representation novelty
              ↓
Calibrate rare-event behavior statistically
              ↓
Aggregate evidence across distributed environments
```

The long-term objective is a detection architecture that is:

**Temporal + Open-Set + Statistically Calibrated + Federated**

rather than simply another closed-set attack classifier.

---

# Current Status

**Status:** Research Prototype — Neural Backbone Complete

Current implementation:

```text
Dataset
   ↓
Leakage-aware preprocessing
   ↓
LOFO split
   ↓
Temporal windows
   ↓
BiLSTM
   ↓
Temporal Attention
   ↓
64-D Latent Space
   ↓
LSTM Decoder
   ↓
Reconstruction Error
   ↓
Latent + Attention Analysis
```

Planned:

```text
Reconstruction Error
        +
Prototype Distance
        ↓
Fused Novelty
        ↓
EVT Calibration
        ↓
Federated Tail Aggregation
        ↓
Open-Set Zero-Day Detection
```

---

# Author

**Saptarshi Mondal**

Computer Science & Engineering — AI & ML

Research interests:

- Artificial Intelligence
- Deep Learning
- Cybersecurity
- Anomaly Detection
- Open-Set Recognition
- Federated Learning
- Extreme Value Theory
- Production ML Systems

---

# Citation

If this work is used in research, experiments, or derivative implementations, please cite the corresponding project/research work.

```bibtex
@software{fed_evt_osad,
  title  = {FedEVT-OSAD: Federated Extreme-Value Open-Set Anomaly Detection},
  author = {Saptarshi Mondal},
  year   = {2026},
  note   = {BiLSTM, Temporal Attention and Autoencoder backbone for zero-day anomaly detection}
}
```

---

## Final Takeaway

FedEVT-OSAD is not designed merely to maximize accuracy on a fixed attack taxonomy.

Its central engineering question is more difficult:

> **Can a model learn a stable representation of network behavior that remains useful when the attack family itself has never been observed before?**

The current BiLSTM-Attention-AE establishes the representation-learning foundation for answering that question. The observed WannaCry result also provides a concrete direction for the next iteration: **combine reconstruction behavior with latent-space novelty and statistically calibrated EVT tails, then extend the detector through federated evidence aggregation.**
