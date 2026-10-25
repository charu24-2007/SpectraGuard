import pytest
import numpy as np
from backend.services.dataset_service import dataset_service, TOTAL_SAMPLES
from src.models.labels import MODULATION_CLASSES


def test_dataset_info():
    info = dataset_service.info()
    assert info["dataset"] == "RadioML 2018.01A"
    assert info["classes"] == 24
    assert info["samples"] == TOTAL_SAMPLES
    assert len(info["class_labels"]) == 24
    assert info["iq_length"] == 1024


def test_valid_sample_retrieval():
    sample = dataset_service.get_sample(0)
    assert sample["dataset_index"] == 0
    assert isinstance(sample["iq"], np.ndarray)
    assert sample["iq"].shape == (1024, 2)
    assert sample["iq"].dtype == np.float32
    assert sample["ground_truth"] in MODULATION_CLASSES
    assert 0 <= sample["gt_class_id"] < 24
    assert isinstance(sample["snr_db"], float)


def test_index_bounds_validation():
    with pytest.raises(ValueError):
        dataset_service.get_sample(-1)

    with pytest.raises(ValueError):
        dataset_service.get_sample(TOTAL_SAMPLES)

    with pytest.raises(ValueError):
        dataset_service.get_sample(99_999_999)

    with pytest.raises(TypeError):
        dataset_service.get_sample("invalid_index")


def test_test_indices_loaded():
    assert len(dataset_service.test_indices) > 0
    rand_idx = dataset_service.get_random_test_index()
    assert 0 <= rand_idx < TOTAL_SAMPLES
    sample = dataset_service.get_sample(rand_idx)
    assert sample["is_test_sample"] is True
