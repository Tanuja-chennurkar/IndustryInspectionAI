"""
Comprehensive Automated Test Suite for FastAPI Backend Endpoints.
Uses FastAPI TestClient to test all 7 endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["num_categories"] == 12

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 12
    categories = [m["category"] for m in models]
    assert "candle" in categories
    assert "pcb1" in categories

def test_inspect_endpoint():
    # Use existing sample image
    sample_img_path = "ml/data/VisA/candle/Data/Images/Normal/000.JPG"
    if not os.path.exists(sample_img_path):
        pytest.skip("Sample image not available for test")

    with open(sample_img_path, "rb") as f:
        files = {"file": ("test_candle.jpg", f, "image/jpeg")}
        data = {"category": "candle"}
        response = client.post("/api/inspect", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert "inspection_id" in res
    assert res["category"] == "candle"
    assert res["status"] in ["NORMAL", "ANOMALY"]
    assert "anomaly_score" in res
    assert "processing_time_ms" in res
    assert "image_url" in res

def test_history_statistics_defects():
    # 1. Test history list
    hist_resp = client.get("/api/inspections")
    assert hist_resp.status_code == 200
    records = hist_resp.json()
    assert isinstance(records, list)

    # 2. Test statistics
    stat_resp = client.get("/api/statistics")
    assert stat_resp.status_code == 200
    stats = stat_resp.json()
    assert "total_inspections" in stats
    assert "defect_rate_percentage" in stats

    # 3. Test defects
    defects_resp = client.get("/api/defects")
    assert defects_resp.status_code == 200
    defects = defects_resp.json()
    assert isinstance(defects, list)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
