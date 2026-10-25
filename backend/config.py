"""
SpectraGuard Backend Configuration.
Reads from environment variables with sensible production defaults.
"""
import os
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_cors_origins() -> List[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings:
    PROJECT_NAME: str = "SpectraGuard API"
    VERSION: str = "2.1.0"
    DESCRIPTION: str = "Passive RF Spectrum Analysis & Electronic Warfare Awareness Platform"

    PROJECT_ROOT: str = PROJECT_ROOT
    DATASET_PATH: str = os.environ.get(
        "DATASET_PATH",
        os.path.join(PROJECT_ROOT, "data", "raw", "RadioML2018", "GOLD_XYZ_OSC.0001_1024.hdf5"),
    )
    MODELS_DIR: str = os.environ.get(
        "MODELS_DIR",
        os.path.join(PROJECT_ROOT, "outputs", "models"),
    )
    DB_PATH: str = os.environ.get(
        "SPECTRAGUARD_DB_PATH",
        os.path.join(PROJECT_ROOT, "backend", "database", "spectraguard.db"),
    )
    CORS_ORIGINS: List[str] = get_cors_origins()
    DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "ResNet1D")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()
