import pytest
import numpy as np
from backend.services.interference_service import (
    interference_service,
    STATE_NO_EVIDENCE,
    STATE_POSSIBLE,
    STATE_INCONCLUSIVE,
)
from backend.services.dataset_service import dataset_service


def test_simulation_does_not_mutate_original_iq():
    sample = dataset_service.get_sample(100)
    orig_copy = sample["iq"].copy()

    interfered, interf_wf = interference_service.generate_interference(
        sample["iq"],
        interference_type="tone-like",
        strength=1.5,
        params={"frequency": 0.25},
    )

    # Verify original array in sample was unchanged
    np.testing.assert_array_equal(sample["iq"], orig_copy)
    # Verify interfered is mathematically different
    assert not np.array_equal(interfered, orig_copy)


def test_simulation_reproducibility_with_seed():
    dummy = np.random.randn(1024, 2).astype(np.float32)

    inj1 = interference_service.random_injection(seed=42)
    inj2 = interference_service.random_injection(seed=42)
    assert inj1["type"] == inj2["type"]
    assert inj1["strength"] == inj2["strength"]
    assert inj1["params"] == inj2["params"]

    out1, _ = interference_service.generate_interference(dummy, inj1["type"], inj1["strength"], inj1["params"], seed=42)
    out2, _ = interference_service.generate_interference(dummy, inj2["type"], inj2["strength"], inj2["params"], seed=42)
    np.testing.assert_array_almost_equal(out1, out2)


def test_original_signal_assessment_consistency():
    sample = dataset_service.get_sample(500)
    orig_iq = sample["iq"]

    # Standalone assessment
    assessment_1 = interference_service.analyze_signal(orig_iq)

    # Assessment when used as baseline in differential analysis
    interfered, _ = interference_service.generate_interference(
        orig_iq, "sweep-like", 1.2, {"f_start": -0.3, "f_end": 0.3}
    )
    assessment_orig_in_sim = interference_service.analyze_signal(orig_iq)

    assert assessment_1["status"] == assessment_orig_in_sim["status"]
    assert assessment_1["pattern"] == assessment_orig_in_sim["pattern"]
    assert assessment_1["metrics"]["signal_power"] == pytest.approx(assessment_orig_in_sim["metrics"]["signal_power"], rel=1e-5)


def test_independent_detection_states():
    # 1. Pure Gaussian noise should yield 'No strong interference-like evidence'
    rng = np.random.RandomState(123)
    clean_noise = (rng.randn(1024, 2) * 0.1).astype(np.float32)
    res_clean = interference_service.analyze_signal(clean_noise)
    assert res_clean["status"] == STATE_NO_EVIDENCE
    assert "detector_version" in res_clean

    # 2. Add strong synthetic tone -> Should yield 'Possible interference-like activity'
    t = np.arange(1024, dtype=np.float32)
    tone_i = 10.0 * np.cos(2 * np.pi * 0.2 * t)
    tone_q = 10.0 * np.sin(2 * np.pi * 0.2 * t)
    tone_signal = clean_noise + np.stack([tone_i, tone_q], axis=1)
    res_tone = interference_service.analyze_signal(tone_signal)
    assert res_tone["status"] == STATE_POSSIBLE
    assert res_tone["pattern"] == "Tone-like"
    assert len(res_tone["evidence"]) > 0


def test_simulation_history_persistence():
    sim_id = interference_service.save_simulation(
        dataset_index=999,
        interference_type="sweep-like",
        strength=1.5,
        detected_pattern="Sweep-like",
        status=STATE_POSSIBLE,
        params={"f_start": -0.2, "f_end": 0.2},
        seed=101,
        orig_pred="QPSK",
        interf_pred="BPSK",
        orig_conf=0.95,
        interf_conf=0.45,
    )
    assert sim_id.startswith("SIM-")

    fetched = interference_service.get_simulation(sim_id)
    assert fetched is not None
    assert fetched["dataset_index"] == 999
    assert fetched["interference_type"] == "sweep-like"
    assert fetched["seed"] == 101
    assert fetched["orig_pred"] == "QPSK"
    assert fetched["interf_pred"] == "BPSK"
