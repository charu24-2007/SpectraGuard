import pytest
import numpy as np
from backend.services.interference_service import (
    interference_service,
    STATE_NO_EVIDENCE,
    STATE_POSSIBLE,
    STATE_INCONCLUSIVE,
)


def test_feature_extraction_fields():
    dummy = np.random.randn(1024, 2).astype(np.float32)
    feats = interference_service.extract_features(dummy)

    expected_fields = [
        "signal_power",
        "energy",
        "papr",
        "noise_floor",
        "peak_power",
        "peak_to_noise",
        "occupancy",
        "spectral_flatness",
        "energy_variance",
        "freq_variance",
    ]
    for field in expected_fields:
        assert field in feats
        assert np.isfinite(feats[field])


def test_zero_silence_graceful_handling():
    silence = np.zeros((1024, 2), dtype=np.float32)
    feats = interference_service.extract_features(silence)
    assert np.isfinite(feats["signal_power"])
    assert np.isfinite(feats["papr"])
    assert np.isfinite(feats["spectral_flatness"])

    analysis = interference_service.analyze_signal(silence)
    valid_states = [
        STATE_NO_EVIDENCE,
        STATE_INCONCLUSIVE,
        STATE_POSSIBLE,
        "No strong evidence",
        "Possible interference",
    ]
    assert analysis["status"] in valid_states


def test_tone_detection_on_pure_sinusoid():
    # Construct a high-SNR pure sinusoid tone
    N = 1024
    t = np.arange(N)
    f0 = 0.20  # Normalized frequency
    pure_tone = np.exp(1j * 2 * np.pi * f0 * t) + (np.random.randn(N) + 1j * np.random.randn(N)) * 0.001
    iq = np.zeros((N, 2), dtype=np.float32)
    iq[:, 0] = np.real(pure_tone)
    iq[:, 1] = np.imag(pure_tone)

    analysis = interference_service.analyze_signal(iq)
    assert analysis["status"] in [STATE_POSSIBLE, "Possible interference"]
    assert analysis["pattern"] == "Tone-like"
    assert len(analysis["evidence"]) > 0


def test_pulsed_detection_on_burst_signal():
    # Construct repeated high-amplitude burst train
    N = 1024
    burst = np.zeros(N, dtype=np.complex128)
    # Burst 1
    burst[100:150] = 50.0 * (np.random.randn(50) + 1j * np.random.randn(50))
    # Burst 2
    burst[500:550] = 50.0 * (np.random.randn(50) + 1j * np.random.randn(50))
    # Low noise elsewhere
    burst += (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1

    iq = np.zeros((N, 2), dtype=np.float32)
    iq[:, 0] = np.real(burst)
    iq[:, 1] = np.imag(burst)

    analysis = interference_service.analyze_signal(iq)
    assert analysis["status"] in [STATE_POSSIBLE, "Possible interference"]
    assert analysis["pattern"] == "Pulsed"
