"""Standalone inference utilities.

Public API:

    from vit.inference import ViTPredictor, PredictionResult
"""

from __future__ import annotations

from src.vit.inference.predictor import PredictionResult, ViTPredictor

__all__ = ["ViTPredictor", "PredictionResult"]