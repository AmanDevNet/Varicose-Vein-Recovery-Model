import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_predict_flow():
    """Verify end-to-end prediction flow with sample input."""
    sample_input = {
        "age": 35, 
        "gender": "Male", 
        "bmi": 26.5, 
        "pain_level": 6, 
        "swelling_level": 5, 
        "activity_level": 4, 
        "beetroot_intake": "Low",
        "fenugreek_intake": "None", 
        "duration_weeks": 8
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/predict", json=sample_input)
    
    assert response.status_code == 200
    data = response.json()
    
    # Assertions based on requirements
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    assert 1 <= data["recovery_weeks_mean"] <= 52
    assert len(data["shap_values"]) == 5
    
    # Verify scenarios exist
    assert "scenarios" in data
    assert "optimal_intake" in data["scenarios"]
