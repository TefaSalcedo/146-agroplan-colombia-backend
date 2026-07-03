"""
Model Loader - Stub for loading trained ML models.

This module will be updated when the ML team provides trained models (.pkl or .joblib files).
Currently, the mock_predictor.py provides mock predictions.

When models are available:
1. Place .pkl/.joblib files in the models/ directory
2. Update this file to load the models
3. Update mock_predictor.py to use real model inference
"""

from typing import Optional
import joblib
from app.config import get_settings

settings = get_settings()


class ModelLoader:
    """Loads trained ML models for zoning and calendar predictions."""
    
    def __init__(self):
        self.zoning_model = None
        self.calendar_model = None
        self._load_models()
    
    def _load_models(self):
        """Load models from disk if they exist."""
        # TODO: Load zoning model
        # try:
        #     self.zoning_model = joblib.load(f"{settings.ml_data_path}/models/zoning_model.pkl")
        # except FileNotFoundError:
        #     pass
        
        # TODO: Load calendar model
        # try:
        #     self.calendar_model = joblib.load(f"{settings.ml_data_path}/models/calendar_model.pkl")
        # except FileNotFoundError:
        #     pass
        pass
    
    def get_zoning_model(self):
        """Get the loaded zoning model."""
        return self.zoning_model
    
    def get_calendar_model(self):
        """Get the loaded calendar model."""
        return self.calendar_model
    
    def is_zoning_model_loaded(self) -> bool:
        """Check if zoning model is loaded."""
        return self.zoning_model is not None
    
    def is_calendar_model_loaded(self) -> bool:
        """Check if calendar model is loaded."""
        return self.calendar_model is not None


# Singleton instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get the singleton model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
