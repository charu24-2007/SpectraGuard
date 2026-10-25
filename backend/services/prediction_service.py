"""
prediction_service.py — thin wrapper kept for backward compatibility.
All real inference is done via model_service.predict() in routes.py.
This file exists so any legacy import does not break startup,
but contains NO dummy data and NO fake predictions.
"""
# This module is intentionally minimal.
# Use backend.services.model_service.model_service for inference.
