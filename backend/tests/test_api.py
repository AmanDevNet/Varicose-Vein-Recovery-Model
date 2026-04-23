import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify /health returns 200 and success status."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_model_info_accuracy():
    """Verify /model-info returns high accuracy (>0.90)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["random_forest_accuracy"] > 0.90
    assert "training_samples" in data
