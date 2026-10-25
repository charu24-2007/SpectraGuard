"""
AnalysisSessionService: Manages backend-owned active sample state so navigation,
page refreshes, and API calls maintain consistent sample references.
"""
import os
import json
import threading
from typing import Optional, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SESSION_FILE = os.path.join(PROJECT_ROOT, "backend", "active_session.json")


class AnalysisSessionService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._active_sample = None
                    cls._instance._load()
        return cls._instance

    def _load(self):
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, "r", encoding="utf-8") as f:
                    self._active_sample = json.load(f)
            except Exception:
                self._active_sample = None

    def _save(self):
        try:
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(self._active_sample, f, indent=2)
        except Exception as e:
            print(f"[AnalysisSessionService] Warning: Could not save session state: {e}")

    def set_active_sample(self, dataset_index: int, signal_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._active_sample = {
                "dataset_index": int(dataset_index),
                "signal_id": signal_id,
                "metadata": metadata or {},
            }
            self._save()

    def get_active_sample(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._active_sample is None:
                self._load()
            return self._active_sample


session_service = AnalysisSessionService()
