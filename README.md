# SpectraGuard: AI-Assisted Passive RF Spectrum Intelligence Platform

SpectraGuard is a passive, defense-oriented RF spectrum analysis and electronic-warfare awareness platform. It analyzes raw in-phase/quadrature (I/Q) radio signals, classifies modulation schemes using deep learning, extracts digital signal processing (DSP) features, and provides explainable interference indicators for human review.

Passive Spectrum Awareness Only: SpectraGuard is designed for passive signal monitoring, modulation recognition, and analytical decision support. It does not implement RF jamming, active signal disruption, or automated counter-attacks. An anomaly alone is not proof of hostile intent.

## What is included

- Deep-learning modulation classification using CNN, ResNet1D, and SpectraNet.
- RadioML 2018.01A dataset integration with partial HDF5 reads.
- A FastAPI backend for inference, DSP analysis, simulations, and history.
- A React + Vite frontend for signal and interference analysis.
- Explainable, rule-based interference indicators.
- Controlled in-memory synthetic interference simulations.
- SQLite-backed analysis history and active-sample synchronization.
- Trained model checkpoints and evaluation results.
- Automated tests for dataset handling, models, DSP, interference, and API routes.
- Dockerfiles and Docker Compose configuration.

## Repository structure

| Path | Purpose |
|---|---|
| [`backend/`](backend/) | FastAPI application and backend services |
| [`backend/api/`](backend/api/) | API routes and request/response schemas |
| [`backend/database/`](backend/database/) | SQLite database initialization and persistence |
| [`backend/services/`](backend/services/) | Dataset, model, DSP, interference, session, and reporting services |
| [`frontend/`](frontend/) | React + Vite web application |
| [`frontend/src/pages/`](frontend/src/pages/) | Signal analysis and interference interface pages |
| [`frontend/src/components/`](frontend/src/components/) | Shared interface components |
| [`src/models/`](src/models/) | CNN, ResNet1D, SpectraNet, labels, and prediction utilities |
| [`notebooks/`](notebooks/) | Model training and evaluation notebook |
| [`outputs/models/`](outputs/models/) | Trained model checkpoints |
| [`outputs/results/`](outputs/results/) | Model comparison and per-class evaluation metrics |
| [`tests/`](tests/) | Automated test suite |
| [`data/raw/`](data/raw/) | Local dataset storage; raw dataset files are not tracked |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`.env.example`](.env.example) | Example environment configuration |
| [`Dockerfile.backend`](Dockerfile.backend) | Backend container configuration |
| [`Dockerfile.frontend`](Dockerfile.frontend) | Frontend container configuration |
| [`docker-compose.yml`](docker-compose.yml) | Container orchestration |

## Dataset

SpectraGuard uses the **RadioML 2018.01A** dataset for modulation-classification experiments.

| Dataset property | Details |
|---|---|
| Dataset | RadioML 2018.01A |
| HDF5 file | `GOLD_XYZ_OSC.0001_1024.hdf5` |
| Total samples | 2,555,904 |
| Samples used in reported experiment | 312,000 |
| Modulation classes | 24 |
| SNR selection | At least 5 dB; sampled records ranged from 6 to 30 dB |
| Train / validation / test | 70% / 15% / 15% |
| Independent test samples | 46,800 |

Download the dataset from [DeepSig's dataset page](https://www.deepsig.ai/datasets/).

Place the downloaded file at:

```text
data/raw/RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5
```

If you store it elsewhere, configure `DATASET_PATH` in your local `.env` file. The application reads the required sample slices rather than loading the entire HDF5 dataset into RAM.

## Model experiments

Three neural network architectures were evaluated using the reported experimental setup.

| Model | Accuracy | Precision | Recall | F1-score | Parameters |
|---|---:|---:|---:|---:|---:|
| CNN | 66.47% | 69.16% | 66.47% | 64.31% | 419,992 |
| SpectraNet | 88.90% | 89.67% | 88.90% | 88.43% | 579,288 |
| ResNet1D | **89.99%** | **90.88%** | **89.99%** | **89.82%** | 7,356,184 |

ResNet1D achieved the highest measured test accuracy in this experiment. SpectraNet is the proposed CNN + Transformer hybrid architecture with competitive results and fewer parameters than ResNet1D.

These metrics describe the reported RadioML experiment. They do not establish performance on live RF hardware or every real-world radio environment.

### Trained models and results

| Artifact | Location |
|---|---|
| CNN checkpoint | [`outputs/models/best_cnn_model.pth`](outputs/models/best_cnn_model.pth) |
| ResNet1D checkpoint | [`outputs/models/best_resnet_model.pth`](outputs/models/best_resnet_model.pth) |
| SpectraNet checkpoint | [`outputs/models/best_spectranet_model.pth`](outputs/models/best_spectranet_model.pth) |
| Model comparison | [`outputs/results/model_comparison_results.csv`](outputs/results/model_comparison_results.csv) |
| Detailed evaluation | [`outputs/results/model_comparison_detailed.csv`](outputs/results/model_comparison_detailed.csv) |
| Per-class metrics | [`outputs/results/per_class_metrics.csv`](outputs/results/per_class_metrics.csv) |
| Training notebook | [`notebooks/SpectraGuard_RF_Models.ipynb`](notebooks/SpectraGuard_RF_Models.ipynb) |

### Supported modulation classes

The classifier supports 24 modulation labels:

```text
OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK,
16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM,
128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC,
AM-DSB-SC, FM, GMSK, OQPSK
```


### Processing workflow

1. **Sample selection:** Select a dataset sample by index or use the configured sample-selection workflow.
2. **Partial HDF5 read:** Retrieve the required I/Q signal, label, and SNR records without loading the complete dataset into memory.
3. **Modulation inference:** Run the selected model and obtain the predicted class and probability information.
4. **DSP feature extraction:** Calculate signal power, peak-to-average power ratio (PAPR), spectral flatness, spectral occupancy, and temporal features.
5. **Session synchronization:** Share the active sample context between signal and interference analysis.
6. **Interference simulation:** Generate a synthetic waveform in memory and compare it with the original sample without modifying the source dataset.
7. **History and reporting:** Persist supported analysis records and provide the corresponding history/reporting workflows.

## Technology stack

| Area | Technologies |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic |
| Deep learning | PyTorch, NumPy |
| Signal processing | SciPy, FFT, STFT-based analysis |
| Dataset access | h5py, RadioML 2018.01A |
| Database | SQLite |
| Frontend | React 18, Vite, Tailwind CSS, Lucide React, Recharts, Axios |
| Testing | Pytest, FastAPI TestClient |
| Deployment | Docker, Docker Compose |

## DSP feature extraction and interference analysis

Modulation classification and interference assessment are treated as separate tasks. The DSP engine evaluates signal properties rather than assuming a change in model prediction proves interference.

| Feature | Purpose |
|---|---|
| Signal power | Estimates mean signal power on a logarithmic scale |
| PAPR | Measures peak power relative to average power |
| Spectral flatness | Helps characterize noise-like versus concentrated spectral energy |
| Spectral occupancy | Estimates the proportion of bins exceeding the estimated noise floor |
| STFT temporal dynamics | Measures changes in energy and peak-frequency behavior across windows |

### Explainable assessment categories

- **Possible interference:** Features are consistent with a tone-like, sweep-like, pulsed, or broadband pattern.
- **No strong evidence:** The measured features do not provide strong evidence of the supported interference patterns.
- **Inconclusive:** The available measurements are ambiguous or insufficient for a confident assessment.

These categories are analytical indicators, not proof of emitter identity, location, intent, or operational role.

## API reference

When the backend is running, interactive API documentation is available at:

[`http://localhost:8000/docs`](http://localhost:8000/docs)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend health and service information |
| `GET /api/models` | Available model information |
| `GET /api/dataset/info` | Dataset metadata and sample counts |
| `POST /api/signals/analyze` | Analyze a selected dataset sample |
| `GET /api/sample/random` | Retrieve a sample through the configured selection workflow |
| `GET /api/sample/active` | Retrieve the active sample reference |
| `POST /api/interference/simulate` | Run controlled, in-memory interference simulation |
| `GET /api/interference/history` | Retrieve recent interference simulation records |

Refer to the running `/docs` page for exact request schemas, query parameters, and response formats.

## Installation and local setup

### Prerequisites

- Python 3.12+
- Node.js 20+ and npm
- Git
- RadioML 2018.01A dataset file
- Model checkpoints in `outputs/models/`

### 1. Clone the repository

```bash
git clone https://github.com/charu24-2007/SpectraGuard.git
cd SpectraGuard
```

### 2. Configure the environment

Windows Command Prompt:

```cmd
copy .env.example .env
```

PowerShell or Linux/macOS:

```bash
cp .env.example .env
```

Edit `.env` if the dataset is stored elsewhere. Do not commit `.env` files containing private or machine-specific settings.

### 3. Install and start the backend

Windows Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the backend from the repository root:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API documentation: `http://localhost:8000/docs`

### 4. Install and start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`.

Build the frontend for production:

```bash
npm run build
```

## Docker Compose

Docker configuration is provided for containerized execution.

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Stop the services:

```bash
docker compose down
```

Check `docker-compose.yml` and the Dockerfiles for the current volume mounts, environment settings, and service ports. Ensure the dataset is available at the configured mounted path.

## Testing

Run the test suite from the repository root:

```bash
pytest -v
```

Run individual test groups:

```bash
pytest tests/test_dataset.py -v
pytest tests/test_models.py -v
pytest tests/test_dsp.py -v
pytest tests/test_interference.py -v
pytest tests/test_api.py -v
```

The tests cover dataset handling, model loading and inference, DSP behavior, interference simulation, and API routes. Results depend on the local environment and the availability of required data and model files.

## Security and safety assumptions

1. **Passive operation:** The platform analyzes signals and simulations; it does not provide radio transmission or jamming functionality.
2. **Dataset integrity:** The source HDF5 dataset is read-only during analysis; simulations operate on in-memory data.
3. **Input validation:** Dataset indices and API request fields must be validated before processing.
4. **Configuration:** Configure allowed frontend origins through environment settings, especially before exposing the API beyond localhost.
5. **Human review:** Interference indicators support investigation and are not conclusive attribution or intent.
6. **Evidence transparency:** Distinguish recorded dataset samples, simulated interference, and any future live-SDR measurements.
7. **Limitations:** RadioML experiment results do not establish live hardware accuracy, geolocation, or emitter identification.

## Implemented features and future scope

| Feature | Status | Notes |
|---|---|---|
| RadioML HDF5 partial reads | Implemented | Reads individual sample slices |
| CNN, ResNet1D, and SpectraNet | Implemented | Trained checkpoints and evaluation results are included |
| Modulation classification | Implemented | Supports 24 RadioML modulation classes |
| DSP feature extraction | Implemented | Power, PAPR, spectral flatness, occupancy, and temporal features |
| Explainable interference indicators | Implemented | Rule-based assessment using measured features |
| In-memory interference simulation | Implemented | Controlled synthetic patterns for comparison |
| Active-sample synchronization | Implemented | Shares sample context across analysis views |
| SQLite analysis history | Implemented | Persists supported analysis records |
| Live SDR integration | Future scope | Requires integration and validation with physical receiver hardware |
| Open-set unknown-signal detection | Future scope | Requires additional data and evaluation |
| Emitter fingerprinting and geolocation | Future scope | Not established by the current dataset or reported experiments |

## Quick navigation

- [Backend](backend/)
- [Frontend](frontend/)
- [Model source code](src/models/)
- [Training notebook](notebooks/SpectraGuard_RF_Models.ipynb)
- [Trained checkpoints](outputs/models/)
- [Evaluation results](outputs/results/)
- [Automated tests](tests/)
- [Environment example](.env.example)
- [Docker Compose](docker-compose.yml)
- [GitHub repository](https://github.com/charu24-2007/SpectraGuard)

## Dataset and research acknowledgement

This project uses RadioML 2018.01A for modulation-classification experiments. Refer to [DeepSig's dataset page](https://www.deepsig.ai/datasets/) for dataset information and terms of use. Cite the dataset and relevant research when using the project in academic or public work.

---

**SpectraGuard** — passive RF spectrum intelligence for explainable, human-reviewed analysis.
