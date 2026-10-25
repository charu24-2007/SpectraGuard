"""
DatasetService: Efficient, partial HDF5 access to RadioML 2018.01A.
Never loads the 21.4 GB dataset entirely into memory.
"""
import os
import sys
import threading
import numpy as np
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.labels import MODULATION_CLASSES
from backend.config import settings

TEST_INDICES_PATH = os.path.join(os.path.dirname(__file__), "..", "test_indices.npy")
TOTAL_SAMPLES = 2_555_904


class DatasetService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._file = None
                    cls._instance.test_indices = np.array([], dtype=np.int64)
                    cls._instance._hdf5_path = settings.DATASET_PATH
        return cls._instance

    def _open(self):
        if self._file is None:
            with self._lock:
                if self._file is None:
                    if not os.path.exists(self._hdf5_path):
                        raise FileNotFoundError(
                            f"RadioML 2018.01A HDF5 dataset not found at '{self._hdf5_path}'. "
                            f"Check DATASET_PATH in .env or config."
                        )
                    import h5py
                    self._file = h5py.File(self._hdf5_path, "r")

                    ti_path = os.path.abspath(TEST_INDICES_PATH)
                    if os.path.exists(ti_path):
                        self.test_indices = np.load(ti_path).astype(np.int64)
                    else:
                        self.test_indices = np.array([], dtype=np.int64)

                    print(f"[DatasetService] Successfully mounted HDF5: {self._hdf5_path}")
                    print(f"[DatasetService] Verified test indices loaded: {len(self.test_indices):,}")

    def validate_index(self, index: int):
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Sample index must be an integer, got {type(index).__name__}")
        if index < 0 or index >= TOTAL_SAMPLES:
            raise ValueError(f"Sample index {index} out of range [0, {TOTAL_SAMPLES - 1}]")

    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single I/Q record from the HDF5 file via slice read.
        Never loads unneeded records into memory.
        """
        self.validate_index(index)
        self._open()

        # Slice partial reads
        iq = np.array(self._file["X"][index], dtype=np.float32)        # (1024, 2)
        y_onehot = np.array(self._file["Y"][index], dtype=np.float32)  # (24,)
        snr = float(self._file["Z"][index][0])                          # scalar dB

        class_idx = int(np.argmax(y_onehot))
        ground_truth = MODULATION_CLASSES[class_idx]

        is_test = False
        if len(self.test_indices) > 0:
            # Binary search or membership check
            is_test = int(index) in self.test_indices

        return {
            "dataset_index": int(index),
            "iq": iq,
            "ground_truth": ground_truth,
            "gt_class_id": class_idx,
            "snr_db": snr,
            "is_test_sample": bool(is_test),
        }

    def get_random_test_index(self) -> int:
        self._open()
        if len(self.test_indices) == 0:
            raise RuntimeError("Test indices not loaded — check backend/test_indices.npy")
        return int(np.random.choice(self.test_indices))

    def search_sample(self, modulation: Optional[str] = None, min_snr: Optional[float] = None) -> int:
        self._open()
        if len(self.test_indices) == 0:
            raise RuntimeError("Test indices not loaded")

        mod_idx = None
        if modulation and modulation in MODULATION_CLASSES:
            mod_idx = MODULATION_CLASSES.index(modulation)

        # Probing test set
        for _ in range(5000):
            idx = int(np.random.choice(self.test_indices))
            if mod_idx is not None:
                y = self._file["Y"][idx]
                if np.argmax(y) != mod_idx:
                    continue
            if min_snr is not None:
                z = float(self._file["Z"][idx][0])
                if z < min_snr:
                    continue
            return idx

        raise ValueError(f"Could not find a sample matching modulation={modulation}, min_snr={min_snr}")

    def info(self) -> Dict[str, Any]:
        self._open()
        return {
            "dataset": "RadioML 2018.01A",
            "name": "RadioML 2018.01A",
            "samples": int(self._file["X"].shape[0]),
            "sample_shape": list(self._file["X"].shape[1:]),
            "classes": len(MODULATION_CLASSES),
            "class_labels": MODULATION_CLASSES,
            "iq_length": 1024,
            "snr_levels": 26,
            "test_samples": int(len(self.test_indices)),
            "split": "70% train / 15% val / 15% test (seed=42)",
            "data_source": "RadioML 2018.01A",
            "hdf5_path": self._hdf5_path,
        }


dataset_service = DatasetService()
