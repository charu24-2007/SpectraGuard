"""
ModelService: Managed model registry for SpectraGuard RF Modulation Classification.
Loads and caches ResNet1D, CNN, and SpectraNet models from checkpoints once at startup.
Uses exact architectures and preprocessing from notebooks/SpectraGuard_RF_Models.ipynb.
"""
import os
import sys
import threading
import torch
import numpy as np
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ResNet1D, CNN, SpectraNet, MODULATION_CLASSES

MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
NUM_CLASSES = len(MODULATION_CLASSES)  # 24

MODEL_CONFIGS = {
    "ResNet1D": {
        "class": ResNet1D,
        "filename": "best_resnet_model.pth",
        "description": "1D Residual Network (Winner - 89.99% Accuracy)",
        "parameters": 7_356_184,
        "test_accuracy": 0.899893,
        "f1_score": 0.898240,
    },
    "SpectraNet": {
        "class": SpectraNet,
        "filename": "best_spectranet_model.pth",
        "description": "CNN-Transformer Hybrid (88.90% Accuracy)",
        "parameters": 579_288,
        "test_accuracy": 0.888974,
        "f1_score": 0.884294,
    },
    "CNN": {
        "class": CNN,
        "filename": "best_cnn_model.pth",
        "description": "Baseline 1D Convolutional Neural Network (66.47% Accuracy)",
        "parameters": 419_992,
        "test_accuracy": 0.664722,
        "f1_score": 0.643075,
    },
}

DEFAULT_MODEL = "ResNet1D"


class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
                    cls._instance._loaded = False
                    cls._instance.classes = MODULATION_CLASSES
                    cls._instance.default_model_name = DEFAULT_MODEL
        return cls._instance

    def load(self, model_names: Optional[List[str]] = None):
        """
        Loads specified models or all available models if model_names is None.
        Loads weights strictly into eval mode on CPU (or CUDA if available).
        """
        targets = model_names or list(MODEL_CONFIGS.keys())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for name in targets:
            if name not in MODEL_CONFIGS:
                print(f"[ModelService] Warning: Unknown model '{name}', skipping.")
                continue

            if name in self._models:
                continue

            cfg = MODEL_CONFIGS[name]
            path = os.path.join(MODELS_DIR, cfg["filename"])
            if not os.path.exists(path):
                print(f"[ModelService] Warning: Checkpoint not found at {path}")
                continue

            try:
                model_inst = cfg["class"](num_classes=NUM_CLASSES)
                state_dict = torch.load(path, map_location=device, weights_only=True)
                model_inst.load_state_dict(state_dict)
                model_inst.to(device)
                model_inst.eval()
                self._models[name] = model_inst
                n_params = sum(p.numel() for p in model_inst.parameters())
                print(f"[ModelService] Loaded {name} ({n_params:,} parameters) on {device}")
            except Exception as e:
                print(f"[ModelService] Error loading {name} from {path}: {e}")

        self._loaded = len(self._models) > 0

    def get_model(self, model_name: Optional[str] = None) -> torch.nn.Module:
        name = model_name or self.default_model_name
        # Case-insensitive resolution
        resolved_name = None
        for k in MODEL_CONFIGS.keys():
            if k.lower() == name.lower():
                resolved_name = k
                break

        if not resolved_name:
            resolved_name = self.default_model_name

        if resolved_name not in self._models:
            # Attempt on-demand load
            self.load([resolved_name])

        if resolved_name not in self._models:
            # Fallback to any loaded model
            if self._models:
                fallback = next(iter(self._models.keys()))
                print(f"[ModelService] Requested '{name}' unavailable. Falling back to '{fallback}'")
                return self._models[fallback], fallback
            raise RuntimeError(f"No models loaded in ModelService. Checked: {name}")

        return self._models[resolved_name], resolved_name

    def predict(self, iq_sample: np.ndarray, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs modulation classification on a single I/Q sample of shape (1024, 2).

        Preprocessing matches the training notebook:
          (1024, 2) -> transpose -> (2, 1024) -> unsqueeze -> (1, 2, 1024)

        Returns:
          predicted_class: str
          predicted_class_id: int
          confidence: float
          all_probabilities: list of 24 floats
          top_predictions: list of dicts with 'modulation', 'class_id', 'probability'
          model_name: str
        """
        if not self._loaded:
            self.load()

        # Input validation
        if not isinstance(iq_sample, np.ndarray):
            iq_sample = np.asarray(iq_sample, dtype=np.float32)

        if iq_sample.shape != (1024, 2):
            raise ValueError(f"Expected I/Q shape (1024, 2), got {iq_sample.shape}")

        if not np.all(np.isfinite(iq_sample)):
            raise ValueError("I/Q sample contains NaN or Inf values")

        model, active_model_name = self.get_model(model_name)
        device = next(model.parameters()).device

        # Preprocessing: (1024, 2) -> (2, 1024) -> (1, 2, 1024)
        tensor = torch.from_numpy(iq_sample.T).unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_class = self.classes[pred_idx]

        top_indices = np.argsort(probs)[::-1][:5]
        top_predictions = [
            {
                "modulation": self.classes[i],
                "class_id": int(i),
                "probability": round(float(probs[i]) * 100, 3),
            }
            for i in top_indices
        ]

        return {
            "predicted_class": predicted_class,
            "predicted_class_id": pred_idx,
            "confidence": confidence,
            "all_probabilities": probs.tolist(),
            "top_predictions": top_predictions,
            "model_name": active_model_name,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns details of all supported models and their current load state."""
        models_info = []
        for name, cfg in MODEL_CONFIGS.items():
            models_info.append({
                "name": name,
                "loaded": name in self._models,
                "description": cfg["description"],
                "parameters": cfg["parameters"],
                "test_accuracy": cfg["test_accuracy"],
                "f1_score": cfg["f1_score"],
                "is_default": (name == self.default_model_name),
            })
        return models_info

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "active_models": list(self._models.keys()),
            "default_model": self.default_model_name,
            "loaded": self._loaded,
            "total_classes": NUM_CLASSES,
            "classes": self.classes,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
        }


# Singleton instance
model_service = ModelService()
