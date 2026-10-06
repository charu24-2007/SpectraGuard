SpectraGuard - AI-Powered Electronic Warfare Awareness & RF Spectrum Intelligence Platform

Abstract

SpectraGuard is a passive, AI-powered Electronic Warfare (EW) awareness and RF spectrum intelligence platform** designed to support defence-oriented electromagnetic-spectrum monitoring. The system processes RF/IQ data using signal processing and artificial intelligence to detect spectrum activity, classify known signal modulations, identify previously unseen or anomalous RF patterns, and analyze interference such as jamming-like signal patterns. Advanced components can further support emitter fingerprinting, behavioural tracking, and simulated multi-receiver geolocation. The project correlates with the defence and electronic-warfare domain because its primary purpose is electromagnetic-spectrum awareness: helping an analyst understand what signals are present, which activity is unusual, whether interference is affecting the spectrum, and where an emitter may be located. These capabilities support passive electronic-support and spectrum-awareness workflows without transmitting, jamming, or performing autonomous countermeasures.SpectraGuard follows a human-in-the-loop approach. AI models generate detections, classifications, anomaly scores, and supporting evidence, while the human analyst reviews the information and makes the final decision. The project uses publicly available research datasets and references and does not claim classified or operational military capability.

Dataset

RadioML 2018.01A - https://www.deepsig.ai/datasets/
Kaggle Mirror - https://www.kaggle.com/datasets/pinxau1000/radioml2018

The dataset contains 24 digital and analog modulation classes with RF/IQ samples across multiple signal-to-noise ratio (SNR) conditions.

Additional Datasets / Data Sources

TorchSig / Sig53 - RF signal generation, classification and wideband experiments
ORACLE - RF/device fingerprinting research
Synthetic interference - generated specifically for controlled jamming/interference detection experiments
TEXBAT - optional future GNSS spoofing research

Research Papers

| # | Title | Source | Link |
|---|-------|--------|------|
| 1 | Over-the-Air Deep Learning Based Radio Signal Classification | IEEE Xplore | https://ieeexplore.ieee.org/document/8267032/ |
| 2 | A Deep Learning Framework for Signal Detection and Modulation Classification | Sensors | https://www.mdpi.com/1424-8220/19/18/4042 |
| 3 | Unsupervised Detection of Offending Radio Transmissions by Means of a Deep Learning Autoencoder | Springer | https://link.springer.com/article/10.1186/s13638-025-02496-3 |
| 4 | Fine-Grained Open Set Signal Modulation Classification via Self-Supervised Pre-Training | IEEE Xplore | https://ieeexplore.ieee.org/document/10918835/ |
| 5 | RF-Enabled Deep-Learning-Assisted Drone Detection and Identification: An End-to-End Approach | MDPI | https://www.mdpi.com/1424-8220/23/9/4202 |

What is Included

- RF/IQ signal classification using deep learning
- RadioML 2018.01A dataset-based experiments
- CNN baseline model
- ResNet model
- Proposed CNN + Transformer SpectraNet model
- Model comparison and evaluation results
- Trained model checkpoints
- Per-class classification metrics
- Streamlit-based reviewer MVP — in development
- Research references related to RF signal intelligence and modulation classification

Repository Structure

| Path | Purpose |
|------|---------|
| `notebooks/` | Model training and evaluation notebooks |
| `outputs/models/` | Trained CNN, ResNet and SpectraNet models |
| `outputs/results/` | Model comparison and per-class evaluation results |
| `data/raw/` | Local RF dataset storage |
| `src/` | Application source code |

Current Experiment

| Parameter | Value |
|-----------|-------|
| Dataset | RadioML 2018.01A |
| Modulation classes | 24 |
| Samples used | 312,000 |
| SNR threshold | ≥ 5 dB |
| Actual SNR range | 6–30 dB |
| Train / Validation / Test | 70% / 15% / 15% |
| Test samples | 46,800 |

Model Experiments

Three architectures were trained and evaluated using the same experimental setup.

1. CNN - baseline convolutional neural network
2. ResNet - residual convolutional neural network
3. SpectraNet - proposed CNN + Transformer architecture

Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|-----------|
| CNN | 66.47% | 69.16% | 66.47% | 64.31% | 419,992 |
| SpectraNet | 88.90% | 89.67% | 88.90% | 88.43% | 579,288 |
| ResNet | 89.99% | 90.88% | 89.99% | 89.82% | 7,356,184 |

ResNet achieved the highest overall performance in the current experiment with 89.99% test accuracy and 89.82% F1-score.

Detailed evaluation results are available in:

- `outputs/results/model_comparison_results.csv`
- `outputs/results/model_comparison_detailed.csv`
- `outputs/results/per_class_metrics.csv`

Planned Capabilities

The RF classification module forms the foundation for additional SpectraGuard capabilities:

- Unknown / open-set signal detection
- RF interference detection and analysis
- Emitter fingerprinting
- Emitter behaviour tracking
- Simulated multi-receiver geolocation
- Evidence-based AI analyst briefing
