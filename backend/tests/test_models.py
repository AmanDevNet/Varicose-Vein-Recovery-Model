import pytest
from models.predict import predictor

def test_models_loaded():
    """Verify that all ML models are loaded correctly."""
    assert predictor.risk_clf is not None, "Ensemble Classifier should be loaded"
    assert predictor.recovery_reg is not None, "Recovery Regressor should be loaded"
    assert predictor.explainer is not None, "SHAP Explainer should be initialized"

def test_model_version():
    """Verify model metadata."""
    # This assumes we have a way to check version, for now just placeholder
    assert hasattr(predictor, 'risk_clf')
