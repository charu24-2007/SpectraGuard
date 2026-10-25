"""
Pydantic Schemas for SpectraGuard API.
Ensures strict type validation, descriptive errors, and clean documentation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

TOTAL_DATASET_SAMPLES = 2_555_904


class AnalyzeRequest(BaseModel):
    dataset_index: int = Field(
        ...,
        ge=0,
        lt=TOTAL_DATASET_SAMPLES,
        description=f"RadioML 2018.01A HDF5 sample index [0 - {TOTAL_DATASET_SAMPLES - 1}]",
        examples=[14520],
    )
    model_name: Optional[str] = Field(
        default="ResNet1D",
        description="Model architecture to use: 'ResNet1D', 'SpectraNet', or 'CNN'",
        examples=["ResNet1D"],
    )


class AutoSimulateRequest(BaseModel):
    dataset_index: Optional[int] = Field(
        default=None,
        ge=0,
        lt=TOTAL_DATASET_SAMPLES,
        description="Dataset index to simulate against. If omitted, uses current active sample from session.",
        examples=[14520],
    )
    model_name: Optional[str] = Field(
        default="ResNet1D",
        description="Model architecture to evaluate: 'ResNet1D', 'SpectraNet', or 'CNN'",
        examples=["ResNet1D"],
    )
    interference_type: Optional[str] = Field(
        default=None,
        description="Optional explicit pattern: 'tone-like', 'sweep-like', 'pulsed', or 'broadband-noise-like'. If omitted, randomly chosen.",
        examples=["sweep-like"],
    )
    strength: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=5.0,
        description="Optional amplitude scaling factor relative to clean signal power",
        examples=[1.2],
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional pattern-specific parameters (e.g. frequency, duration, interval, bandwidth)",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic simulation and reproducibility",
        examples=[42],
    )


class ActiveSampleSetRequest(BaseModel):
    dataset_index: int = Field(
        ...,
        ge=0,
        lt=TOTAL_DATASET_SAMPLES,
        description="Dataset index to set as active across the application",
    )
    signal_id: Optional[str] = Field(
        default=None,
        description="Optional signal ID associated with the analysis",
    )


class TopPredictionItem(BaseModel):
    modulation: str
    class_id: int
    probability: float


class ModulationResult(BaseModel):
    predicted: str
    confidence: float
    top5: List[TopPredictionItem]


class DspMetrics(BaseModel):
    signal_power: float
    energy: float
    papr: float
    noise_floor: float
    peak_power: float
    peak_to_noise: float
    occupancy: float
    spectral_flatness: float
    energy_variance: float
    freq_variance: float


class DspAssessment(BaseModel):
    status: str
    pattern: str
    evidence: List[str]
    explanation: str


class WaveformData(BaseModel):
    i: List[float]
    q: List[float]


class SpectrumData(BaseModel):
    frequency: List[float]
    magnitude: List[float]
    freq_label: str
    mag_label: str


class SpectrogramData(BaseModel):
    time: List[float]
    frequency: List[float]
    power: List[List[float]]
    freq_label: str
    time_label: str


class SignalAnalysisResponse(BaseModel):
    signal_id: str
    dataset_index: int
    is_test_sample: bool
    ground_truth_modulation: str
    snr_db: float
    predicted_modulation: str
    confidence: float
    correct_prediction: bool
    model: str
    data_source: str
    top_predictions: List[TopPredictionItem]
    waveform: WaveformData
    spectrum: SpectrumData
    spectrogram: SpectrogramData
    interference: Dict[str, Any]


class ModelInfoItem(BaseModel):
    name: str
    loaded: bool
    description: str
    parameters: int
    test_accuracy: float
    f1_score: float
    is_default: bool


class PaginatedSignalsResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    pages: int


class PaginatedSimulationsResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    pages: int
