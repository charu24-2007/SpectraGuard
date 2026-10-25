import pytest
import torch
import numpy as np
from backend.services.model_service import model_service, MODEL_CONFIGS
from src.models import ResNet1D, CNN, SpectraNet, MODULATION_CLASSES


def test_models_loading_and_parameter_counts():
    model_service.load()

    # 1. ResNet1D
    resnet, r_name = model_service.get_model("ResNet1D")
    assert r_name == "ResNet1D"
    r_params = sum(p.numel() for p in resnet.parameters())
    assert r_params == 7_356_184

    # 2. CNN
    cnn, c_name = model_service.get_model("CNN")
    assert c_name == "CNN"
    c_params = sum(p.numel() for p in cnn.parameters())
    assert c_params == 419_992

    # 3. SpectraNet
    spectranet, s_name = model_service.get_model("SpectraNet")
    assert s_name == "SpectraNet"
    s_params = sum(p.numel() for p in spectranet.parameters())
    assert s_params == 579_288


def test_model_listing():
    models = model_service.list_models()
    assert len(models) == 3
    names = [m["name"] for m in models]
    assert "ResNet1D" in names
    assert "SpectraNet" in names
    assert "CNN" in names


def test_inference_shape_and_probabilities():
    dummy = np.random.randn(1024, 2).astype(np.float32)

    for m in ["ResNet1D", "SpectraNet", "CNN"]:
        res = model_service.predict(dummy, model_name=m)
        assert res["predicted_class"] in MODULATION_CLASSES
        assert 0 <= res["predicted_class_id"] < 24
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["all_probabilities"]) == 24
        assert abs(sum(res["all_probabilities"]) - 1.0) < 1e-4

        # Top 5 check
        top5 = res["top_predictions"]
        assert len(top5) == 5
        assert top5[0]["modulation"] == res["predicted_class"]
        # Monotonically descending probability check
        probs = [p["probability"] for p in top5]
        assert probs == sorted(probs, reverse=True)


def test_inference_input_validation():
    # Bad shape
    with pytest.raises(ValueError, match="Expected I/Q shape"):
        model_service.predict(np.zeros((512, 2), dtype=np.float32))

    # NaN input
    bad_nan = np.zeros((1024, 2), dtype=np.float32)
    bad_nan[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model_service.predict(bad_nan)

    # Inf input
    bad_inf = np.zeros((1024, 2), dtype=np.float32)
    bad_inf[10, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        model_service.predict(bad_inf)
