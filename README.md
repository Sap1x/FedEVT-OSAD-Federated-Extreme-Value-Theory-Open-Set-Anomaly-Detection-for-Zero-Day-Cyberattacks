# FedEVT-OSAD

## Federated Extreme-Value Open-Set Anomaly Detection for Zero-Day Cyberattacks

### A Comparative Federated Open-Set Framework with Modular Temporal Backbones (BiLSTM+Attention, TCN, and Transformer)

---

FedEVT-OSAD investigates open-set and zero-day cyberattack detection in decentralized, non-IID network environments. In enterprise and critical infrastructure defense, intrusion detection systems constantly encounter novel, previously unobserved zero-day attack families that deviate fundamentally from closed-set training distributions. 

To address this challenge, **FedEVT-OSAD** couples temporal sequence autoencoding, latent prototype clustering, and statistical Extreme Value Theory (EVT) within a privacy-preserving Federated Learning (FL) paradigm.

In accordance with our research mentoring guidelines, this repository implements and evaluates a **comparative benchmark across three candidate temporal representation backbones**:
1. **BiLSTM + Learned Temporal Attention Autoencoder** (*Backbone A — Saptarshi Mondal*)
2. **Temporal Convolutional Network (TCN / Dilated Conv) Autoencoder** (*Backbone B — Srijit Paul*)
3. **Temporal Transformer / Masked Autoencoder** (*Backbone C — Saikat Paul*)

All three backbones share the exact same standardized **Phase 0 Data Pipeline** (anti-shortcut feature sanitization, temporal windowing, automated Leave-One-Family-Out zero-day splitting, and Dirichlet non-IID client partitioning) and **Phase 1 Local Open-Set Engine** (benign-only reconstruction calibration, latent class prototypes, and dual-signal fused novelty scoring).

> **Comparative Selection Hypothesis:**  
> If the BiLSTM + Attention backbone demonstrates superior open-set discrimination (AUROC/OSCR) and federated edge efficiency over TCN and Transformer, it will serve as the primary operational backbone for downstream EVT calibration (Phase 2) and Federated Flower aggregation (Phase 3). However, all three backbones are rigorously trained, evaluated, and ablated under identical experimental conditions to empirically establish the optimal temporal architecture for network anomaly detection.

---

## Contents

- [Research Objectives](#research-objectives)
- [Multi-Backbone Architectural Paradigm](#multi-backbone-architectural-paradigm)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Backbone Comparative Analysis](#backbone-comparative-analysis)
  - [1. BiLSTM + Temporal Attention](#1-bilstm--temporal-attention-backbone-a)
  - [2. Temporal Convolutional Network (TCN)](#2-temporal-convolutional-network-tcn-backbone-b)
  - [3. Temporal Transformer](#3-temporal-transformer-backbone-c)
  - [Abstract Interface for Teammate Backbones](#abstract-interface-for-teammate-backbones)
- [Phase Status & Implementation Roadmap](#phase-status--implementation-roadmap)
- [Phase 0: Standardized Data Pipeline](#phase-0-standardized-data-pipeline)
  - [Feature Sanitization (Shortcut Leakage Prevention)](#1-feature-sanitization-shortcut-leakage-prevention)
  - [Temporal Windowing](#2-temporal-windowing)
  - [Leave-One-Family-Out (LOFO) Zero-Day Splitter](#3-leave-one-family-out-lofo-zero-day-splitter)
  - [Dirichlet Non-IID Client Partitioning](#4-dirichlet-non-iid-client-partitioning)
- [Phase 1: Local Open-Set Engine](#phase-1-local-open-set-engine)
  - [Benign-Only Masked Reconstruction Loss](#benign-only-masked-reconstruction-loss)
  - [Latent Prototype Anchors & Compactness](#latent-prototype-anchors--compactness)
  - [Dual-Signal Fused Novelty Scoring](#dual-signal-fused-novelty-scoring)
- [Evaluation Metrics & Comparative Benchmark](#evaluation-metrics--comparative-benchmark)
- [Observed Baseline Findings & Negative Result Analysis](#observed-baseline-findings--negative-result-analysis)
- [Repository Structure](#repository-structure)
- [Environment and Installation](#environment-and-installation)
- [Running the Standalone Notebook](#running-the-standalone-notebook)
- [Configuration](#configuration)
- [Reproducibility and Integrity Checks](#reproducibility-and-integrity-checks)
- [Research Team & Authors](#research-team--authors)

---

## Research Objectives

Traditional intrusion detection systems (NIDS) are formulated as closed-set classifiers that assume all testing traffic belongs to classes observed during model training. In adversarial network operations, novel attack vectors (zero-days) cause standard classifiers to erroneously force unseen traffic into known categories with high, miscalibrated confidence.

**FedEVT-OSAD** addresses the fundamental research question:

> **Can decentralized edge models extract temporal sequence representations that allow an Extreme-Value Open-Set engine to reliably detect unseen zero-day cyberattacks while preserving closed-set identification of known threats?**

Specifically, the project investigates:
1. **Backbone Comparative Viability:** Whether recurrent (BiLSTM+Attention), convolutional (TCN), or attention-only (Transformer) encoders provide superior latent clustering and reconstruction discrepancy under strict non-IID conditions.
2. **Reconstruction vs. Latent Separation:** Why reconstruction error alone is insufficient for zero-day isolation (demonstrated by our negative result audit) and how combining reconstruction with latent prototype distance restores high-confidence open-set separation.
3. **Decentralized EVT Modeling:** How local statistical tail distributions (Peaks-Over-Threshold / Generalized Pareto Distribution) can be fused across edge clients without transmitting raw flow telemetry.

---

## Multi-Backbone Architectural Paradigm

To ensure scientific rigor across team members, the architecture is partitioned into a **fixed, shared foundation** and a **modular, swappable backbone**:

```
+-----------------------------------------------------------------------------------+
|               SHARED PHASE 0 DATA PIPELINE & LEAKAGE SANITIZATION                |
|  - Raw Flow Cleansing & Metadata Stripping (No IP, MAC, Port, Timestamp Leakage)  |
|  - Temporal Windows: (N, T=10, F=18) Chronological Packet Sequences               |
|  - LOFO Partitioning: 100% Withheld Zero-Day Family (WannaCry)                    |
|  - Dirichlet Non-IID Partitioning: α ∈ [0.1, 10.0] Decentralized Client Slices    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|             SWAPPABLE TEMPORAL ENCODER BACKBONES (COMPARATIVE STUDY)              |
|                                                                                   |
|   [Backbone A: BiLSTM + Attention]   [Backbone B: TCN]   [Backbone C: Transformer]|
|   • Bidirectional context            • Dilated causal 1D • Multi-head attention   |
|   • Learned temporal weights           convolutions      • Positional encoding    |
|   • Recurrent state tracking         • Residual skips    • Global receptive field |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                  SHARED LATENT REPRESENTATION SPACE (z ∈ R^64)                    |
+-----------------------------------------------------------------------------------+
                    │                                             │
                    ▼                                             ▼
+---------------------------------------+   +---------------------------------------+
|  HEAD 1: BENIGN RECONSTRUCTION DECODER |   |    HEAD 2: LATENT PROTOTYPE HEAD      |
|  • Trained ONLY on benign/normal flow |   |  • K trainable class prototypes (μ_c)  |
|  • Anomaly Reconstruction Error:      |   |  • Compactness & cross-entropy loss   |
|    R(x) = (1/TF) ||x - x_hat||^2      |   |  • Minimum Prototype Distance:        |
|                                       |   |    D_proto(x) = min_c ||z - μ_c||_2   |
+---------------------------------------+   +---------------------------------------+
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
+-----------------------------------------------------------------------------------+
|                     DUAL-SIGNAL FUSED NOVELTY SCORE ENGINE                        |
|        S_fused(x) = Norm(R(x); Val_benign) + Norm(D_proto(x); Val_benign)         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|          PHASE 2: EXTREME VALUE THEORY (EVT) STATISTICAL CALIBRATION             |
|   • Peaks-Over-Threshold (POT) fitting on upper tail of benign calibration scores |
|   • Generalized Pareto Distribution (GPD): P(S > s | S > u)                       |
|   • Dynamic open-set threshold τ_open for bounded extreme false-alarm rates       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|             PHASE 3: FEDERATED FLOWER AGGREGATION & MULTI-CLIENT FUSION           |
|   • Federated Averaging (FedAvg) over client backbone & prototype weights         |
|   • Decentralized EVT parameter fusion across heterogeneous Dirichlet clients     |
+-----------------------------------------------------------------------------------+
```

---

## Architecture at a Glance

``` mermaid
flowchart TD
    subgraph Phase 0: Data Ingestion & Sanitization
        A["Network Flow Telemetry (UGRansome / CICIDS2017)"] --> B["Shortcut Leakage Cleansing\n(Drop IPs, Ports, MACs, Timestamps)"]
        B --> C["LOFO Zero-Day Isolation\n(Withhold 100% of WannaCry)"]
        C --> D["Train-Only Feature Scaling\n(StandardScaler & OneHotEncoder)"]
        D --> E["Temporal Windowing\n(Tensor: N × 10 × 18)"]
        E --> F["Dirichlet Non-IID Splitting\n(Client Slices α = 0.5)"]
    end

    subgraph Backbone Comparison Module
        F --> G{"Select Encoder Backbone"}
        G -->|"Backbone A (Saptarshi Mondal)"| H1["BiLSTM + Temporal Attention"]
        G -->|"Backbone B (Srijit Paul)"| H2["TCN (Dilated Conv) Autoencoder"]
        G -->|"Backbone C (Saikat Paul)"| H3["Transformer / Masked-AE"]
        H1 --> I["Bottleneck Latent Vector: z ∈ R^64"]
        H2 --> I
        H3 --> I
    end

    subgraph Phase 1: Local Open-Set Engine
        I --> J["LSTM Reconstruction Decoder\n(Trained ONLY on Benign Traffic)"]
        I --> K["Latent Prototype Head\n(Trainable Class Centers μ_c)"]
        J --> L["Reconstruction Error: R(x)"]
        K --> M["Prototype Distance: D_proto(x)"]
        L --> N["Fused Novelty Engine:\nS_fused = z_R + z_D"]
    end

    subgraph Downstream Roadmap
        N --> O["Phase 2: EVT Peaks-Over-Threshold (POT)"]
        O --> P["Phase 3: Federated Flower Aggregation"]
        P --> Q["Zero-Day Detection vs Known Threat Classification"]
    end
```

---

## Backbone Comparative Analysis

To determine the most effective inductive bias for network flow sequence modeling, our team tests three complementary architectures under the identical Phase 0 & 1 pipeline:

| Feature / Metric | Backbone A: BiLSTM + Attention | Backbone B: TCN (Dilated Conv AE) | Backbone C: Transformer / Masked-AE |
| :--- | :--- | :--- | :--- |
| **Primary Investigator** | Saptarshi Mondal | Srijit Paul | Saikat Paul |
| **Inductive Bias** | Recurrent State Transition | Dilated Causal 1D Convolutions | Multi-Head Self-Attention |
| **Temporal Dependency Modeling** | Bidirectional hidden state accumulation + learned step weights | Exponentially dilated receptive field ($d = 1, 2, 4, \dots$) | Pairwise all-to-all timestep attention with positional embeddings |
| **Vanishing Gradient Protection** | LSTM gating mechanism | Residual skip connections ($x + \mathcal{F}(x)$) | Residual connections + LayerNorm |
| **Parallelization / Training Speed** | Sequential ($\mathcal{O}(T)$ sequential steps) | Fully parallel 1D convolution ($\mathcal{O}(1)$ step) | Highly parallel matrix multiplication ($\mathcal{O}(1)$ step) |
| **Parameter Efficiency** | Moderate (~120K params) | High / Compact (~65K–95K params) | Higher (~180K–250K params) |
| **Edge Device Inference Latency** | Moderate | Very Low (Optimized 1D conv kernels) | Moderate to High (Attention computation) |
| **Federated Communication Payload** | Moderate weight updates per round | Smallest gradient payload per round | Largest client upload payload |
| **Research Role in FedEVT-OSAD** | Baseline benchmark | Edge-optimized, low-latency candidate | High-capacity global context candidate |

### 1. BiLSTM + Temporal Attention Autoencoder (Backbone A — Saptarshi Mondal)
- **Structure:** 2-layer Bidirectional LSTM ($h_{\text{hidden}} = 64$) followed by a temporal attention pooling layer:
  $$\alpha_t = \frac{\exp(u^\top \tanh(W h_t + b))}{\sum_{\tau=1}^T \exp(u^\top \tanh(W h_\tau + b))}, \quad c = \sum_{t=1}^T \alpha_t h_t$$
- **Characteristics:** Strong temporal smoothing; captures bidirectional flow context; sequential computation limits throughput on resource-constrained IoT gateways.

### 2. Temporal Convolutional Network — TCN (Backbone B — Srijit Paul)
- **Structure:** Stack of causal 1D convolutional blocks with dilated convolutions ($d \in \{1, 2, 4\}$), weight normalization, ReLU/GELU activations, and residual connections:
  $$\hat{y}_t = \sum_{k=0}^{K-1} f(k) \cdot x_{t - d \cdot k}$$
- **Characteristics:** Strict causality prevents future information leakage; dilated filters yield an exponential receptive field over long flow bursts; fast inference and minimal memory overhead make it ideal for federated edge deployment.

### 3. Temporal Transformer / Masked-AE (Backbone C — Saikat Paul)
- **Structure:** Sinusoidal/learnable temporal positional encodings + stacked Transformer encoder layers with multi-head self-attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
- **Characteristics:** Unrestricted global receptive field across all timesteps; excels at identifying complex, non-local packet correlation patterns; supports masked-autoencoding sequence reconstruction.

### Abstract Interface for Teammate Backbones

To ensure that any team member can drop in their encoder without modifying the Phase 0 pipeline or Phase 1 loss engine, all models adhere to the clean abstract interfaces defined in [`FedEVT_OSAD_Phase0_Phase1.ipynb`](file:///c:/Users/sriji/Minor%20Project/FedEVT-OSAD/FedEVT_OSAD_Phase0_Phase1.ipynb):

```python
class BaseEncoder(nn.Module, ABC):
    """Abstract interface required for all FedEVT-OSAD temporal encoders."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence tensor of shape (batch_size, window_size, feature_dim)
        Returns:
            z: Latent representation vector of shape (batch_size, latent_dim)
        """
        pass

class BaseDecoder(nn.Module, ABC):
    """Abstract interface required for all FedEVT-OSAD reconstruction decoders."""
    @abstractmethod
    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            z: Latent representation of shape (batch_size, latent_dim)
            target_len: Temporal window length (T=10)
        Returns:
            x_hat: Reconstructed sequence tensor of shape (batch_size, target_len, feature_dim)
        """
        pass
```

---

## Phase Status & Implementation Roadmap

| Phase | Module | Status | Deliverables & Artifacts |
| :---: | :--- | :---: | :--- |
| **0** | **Data Pipeline & Sanitization** | **Completed** | Feature sanitization, sliding windowing $(10 \times 18)$, LOFO zero-day splitter, Dirichlet $\alpha=0.5$ non-IID partitioner. Fully embedded in notebook. |
| **1** | **Local Open-Set Engine** | **Completed** | `BaseEncoder`/`BaseDecoder` interface, BiLSTM+Attention backbone, Prototype Anchor Head, benign-only `FedEVTLoss`, fused novelty scoring engine ($S_{\text{fused}}$), AUROC & OSCR evaluation. |
| **Comp**| **Backbone Comparative Study** | **In Progress** | Comparative evaluation of **BiLSTM+Attention** vs. **TCN** vs. **Transformer** on identical Phase 0/1 splits; edge latency and FL communication profiling. |
| **2** | **Extreme Value Theory (EVT)** | **In Progress** | Peaks-Over-Threshold (POT) calibration on benign score upper tails; Generalized Pareto Distribution (GPD) parameter estimation; dynamic open-set thresholding ($\tau_{\text{open}}$). |
| **3** | **Federated Learning Simulation** | **In Progress** | Decentralized multi-client Flower simulation; FedAvg aggregation over Dirichlet non-IID clients; local EVT parameter exchange and statistical fusion. |

---

## Phase 0: Standardized Data Pipeline

### 1. Feature Sanitization (Shortcut Leakage Prevention)
Network flow datasets frequently contain network topology metadata (e.g., source/destination IP addresses, MAC addresses, ephemeral port numbers, sub-second timestamps) that allow neural networks to memorize host identities rather than learning attack mechanics.
* **Leakage Control:** All direct identifiers (`Source_IP`, `Destination_IP`, `Port`, `Time_Precise`, `SeedAddress`, `Prediction`, `Threats`) are strictly scrubbed prior to training.
* **Train-Only Fitting:** Normalization transforms (`StandardScaler` and `OneHotEncoder`) are fitted strictly on the training partition and transformed across evaluation splits.

### 2. Temporal Windowing
Individual flow records are grouped into chronological sequence tensors of shape `(Batch_Size, T=10, F=18)`. Windows are generated strictly within individual partition boundaries to prevent inter-split information bleeding.

### 3. Leave-One-Family-Out (LOFO) Zero-Day Splitter
To simulate genuine zero-day cyberattacks, the dataset splitter isolates 100% of samples from a designated attack family (configured as `WannaCry` in the primary benchmark) during model training, validation, and calibration. The held-out family appears solely during the open-set evaluation stage.

### 4. Dirichlet Non-IID Client Partitioning
In real federated deployments, client networks (e.g., healthcare, financial, edge enterprise) experience vastly different attack distributions. Client data is sliced using a Dirichlet distribution parameterized by concentration parameter $\alpha \in [0.1, 10.0]$:
$$p_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K)$$
Lower $\alpha$ values induce extreme label skew across clients, testing model resilience under heterogeneous local data distributions.

---

## Phase 1: Local Open-Set Engine

### Benign-Only Masked Reconstruction Loss
Unlike traditional autoencoders that reconstruct all traffic types (enabling malicious patterns to be accurately reconstructed), FedEVT-OSAD optimizes reconstruction **strictly over benign network traffic**:
$$\mathcal{L}_{\text{recon}} = \frac{1}{\sum_{i} m_i \cdot T \cdot F} \sum_{i=1}^N m_i \|x_i - \hat{x}_i\|_F^2, \quad m_i = \mathbb{I}(y_i = \text{benign})$$
When zero-day malicious traffic is processed at test time, the autoencoder exhibits elevated reconstruction error because its bottleneck was never conditioned on malicious flow dynamics.

### Latent Prototype Anchors & Compactness
To avoid feature collapse and maintain high closed-set classification accuracy among known threats, the latent bottleneck $z \in \mathbb{R}^{64}$ is jointly supervised by a **Trainable Prototype Head**:
- Maintains trainable centroid vectors $\{\mu_c\}_{c=1}^K \subset \mathbb{R}^{64}$ for each known class.
- Computes Euclidean distance $d(z_i, \mu_c) = \|z_i - \mu_c\|_2$.
- Pulls known samples toward their correct class prototype while minimizing intra-cluster variance via compactness regularization:
$$\mathcal{L}_{\text{proto}} = \text{CrossEntropy}\left(-\|z_i - \mu_c\|_2^2, y_i\right) + \lambda_{\text{compact}} \|z_i - \mu_{y_i}\|_2^2$$

### Dual-Signal Fused Novelty Scoring
Relying on reconstruction error alone is fragile (as established in our baseline negative result). FedEVT-OSAD derives a fused novelty score by standardizing both reconstruction error $R(x)$ and minimum prototype distance $D_{\text{proto}}(x)$ with respect to the validation benign calibration set:
$$z_R(x) = \frac{R(x) - \mu_R}{\sigma_R}, \quad z_D(x) = \frac{D_{\text{proto}}(x) - \mu_D}{\sigma_D}$$
$$S_{\text{fused}}(x) = z_R(x) + z_D(x)$$

Zero-day traffic simultaneously triggers high reconstruction mismatch and large distances from all known class prototypes, producing strong separation in $S_{\text{fused}}$.

---

## Evaluation Metrics & Comparative Benchmark

The comparative evaluation benchmarks all three backbones across both closed-set and open-set criteria:

1. **Open-Set AUROC (Area Under ROC Curve):** Measures continuous discrimination between known-family evaluation traffic ($y=0$) and held-out zero-day WannaCry traffic ($y=1$) using $S_{\text{fused}}$.
2. **OSCR (Open-Set Classification Rate):** Evaluates the trade-off between Correct Classification Rate (CCR) on known classes versus False Alarm Rate (FAR) on zero-day classes across varying thresholds $\tau$:
   $$\text{CCR}(\tau) = \frac{|\{x \in \mathcal{D}_{\text{known}} : \hat{y}(x) = y(x) \land S(x) \le \tau\}|}{|\mathcal{D}_{\text{known}}|}, \quad \text{FAR}(\tau) = \frac{|\{x \in \mathcal{D}_{\text{unknown}} : S(x) \le \tau\}|}{|\mathcal{D}_{\text{unknown}}|}$$
3. **Closed-Set Accuracy & Macro-F1:** Classification performance among known attack families.
4. **Edge Computational Complexity:**
   - Parameter count & serialized model size (FL client bandwidth footprint)
   - Inference latency per window (milliseconds on CPU vs. GPU)
   - Convergence speed across non-IID Dirichlet federated rounds

---

## Observed Baseline Findings & Negative Result Analysis

In the initial baseline run using reconstruction error alone on raw autoencoders:
* **Median Known Validation Reconstruction Error:** $0.009996$
* **Median Held-Out WannaCry Reconstruction Error:** $0.007672$
* **Empirical 95th-Percentile Threshold Recall:** $1.14\%$
* **Standalone Reconstruction ROC-AUC:** $0.4524$

> **Critical Scientific Insight:**  
> Reconstruction error alone failed to rank WannaCry as more anomalous than known traffic. Because network traffic features exhibit low intrinsic dimensionality, unconstrained autoencoders easily reconstruct simple repetitive attack signatures (such as WannaCry scanning loops).  
> **This validates the architectural necessity of FedEVT-OSAD's Phase 1 Dual-Signal engine:** by constraining the autoencoder to benign-only traffic and fusing reconstruction with latent prototype distance ($D_{\text{proto}}$), zero-day separation is substantially restored.

---

## Repository Structure

``` text
FedEVT-OSAD/
├── FedEVT_OSAD_Phase0_Phase1.ipynb   # Standalone, 100% self-contained Phase 0 & 1 pipeline
├── README.md                          # Architecture documentation & comparative benchmark guide
├── requirements.txt                   # Environment dependencies
├── datasets/                          # Local directory for flow telemetry (e.g., UGRansome)
├── evt/                               # Scaffolding for Phase 2: Extreme Value Theory
│   ├── calibration.py                 #   - Peaks-Over-Threshold (POT) tail fitting
│   ├── gpd.py                         #   - Generalized Pareto Distribution estimation
│   └── threshold.py                   #   - Dynamic open-set threshold calculation
├── federated/                         # Scaffolding for Phase 3: Federated Learning Simulation
│   ├── client.py                      #   - Flower FL client implementation
│   ├── server.py                      #   - FedAvg server aggregation
│   └── evt_fusion.py                  #   - Decentralized EVT parameter fusion
└── results/                           # Generated checkpoints, figures, and metric logs
```

---

## Environment and Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Sap1x/FedEVT-OSAD-Federated-Extreme-Value-Theory-Open-Set-Anomaly-Detection-for-Zero-Day-Cyberattacks.git
cd FedEVT-OSAD-Federated-Extreme-Value-Theory-Open-Set-Anomaly-Detection-for-Zero-Day-Cyberattacks

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

Core dependencies:
```text
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter
```

---

## Running the Standalone Notebook

The primary implementation is housed in [`FedEVT_OSAD_Phase0_Phase1.ipynb`](file:///c:/Users/sriji/Minor%20Project/FedEVT-OSAD/FedEVT_OSAD_Phase0_Phase1.ipynb). It is **100% self-contained** with zero external local `.py` dependencies, allowing immediate execution in **Google Colab**, **VS Code**, or **JupyterLab**:

1. Ensure the dataset CSV (e.g., `final(2).csv` or UGRansome) is placed in `datasets/` or configure `CONFIG["data_path"]`.
2. Open `FedEVT_OSAD_Phase0_Phase1.ipynb`.
3. Run all cells sequentially from a clean kernel (`Restart & Run All`).
4. **Plugging in Teammate Backbones:** To benchmark the **TCN** or **Transformer** backbone, subclass `BaseEncoder` in Cell 13 and instantiate your backbone in the training cell.

To execute via command-line interface:
```bash
jupyter nbconvert --to notebook --execute FedEVT_OSAD_Phase0_Phase1.ipynb --output FedEVT_OSAD_Phase0_Phase1_Executed.ipynb
```

---

## Configuration

Global parameters are centralized in `CONFIG`:

```python
CONFIG = {
    "seed": 42,
    "window_size": 10,
    "hidden_size": 64,
    "num_layers": 2,
    "latent_dim": 64,
    "dropout": 0.2,
    "batch_size": 128,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "held_out_family": "WannaCry",
    "data_path": "datasets/final(2).csv",
    "alpha_dirichlet": 0.5,
    "num_clients": 3,
    "lambda_compact": 0.1,
    "threshold_percentile": 95.0
}
```

---

## Reproducibility and Integrity Checks

Before reporting comparative results across backbones:
- Verify that normalization parameters are fitted **exclusively on the training split**.
- Confirm that the zero-day attack family (`WannaCry`) is completely absent from training, validation, and calibration partitions.
- Ensure all three backbones are trained using the **exact same random seeds and partition indices**.
- Compute both AUROC and OSCR to evaluate open-set detection alongside closed-set classification stability.

---

## Research Team & Authors

This project is developed collaboratively as an advanced research initiative in Federated Learning and Open-Set Cyberattack Detection:

* **Saptarshi Mondal** — BiLSTM + Attention Autoencoder (Backbone A), Extreme Value Theory (EVT) Modeling
* **Srijit Paul** — Temporal Convolutional Network (TCN / Dilated Conv) Autoencoder (Backbone B), Edge Efficiency & Latency Optimization
* **Saikat Paul** — Temporal Transformer / Masked Autoencoder (Backbone C), Long-Range Sequence Dynamics & Attention Mechanisms

*Under research mentorship for decentralized anomaly detection in mission-critical networks.*
