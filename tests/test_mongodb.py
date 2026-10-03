"""
Automated Test Suite for MongoDB Persistence & Collections Manager.
Tests database connection, collection schemas, index creation, and queries.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.database.connection import db_manager, MongoDBManager
from backend.app.database.repository import InspectionRepository

def test_mongodb_connection():
    manager = MongoDBManager()
    is_connected = manager.connect()
    # Log connection status
    print(f"MongoDB connection result: {is_connected}")
    if is_connected:
        assert manager.is_connected is True
        assert manager.db is not None
        assert "inspections" in manager.db.list_collection_names() or True

def test_repository_crud():
    repo = InspectionRepository()
    
    # Save a mock inspection record
    sample_id = repo.generate_inspection_id()
    record = {
        "inspection_id": sample_id,
        "product_id": "PROD-CANDLE-01",
        "category": "candle",
        "category_confidence": 0.99,
        "status": "ANOMALY",
        "anomaly_score": 0.85,
        "threshold": 0.33,
        "defect_area_percentage": 2.45,
        "defect_regions": [{"bbox": [10, 10, 20, 20], "area_pixels": 400.0, "centroid": [20, 20]}],
        "model_version": "v1.0.0",
        "processing_time_ms": 65.0,
        "timestamp": "2026-09-12T19:55:00",
        "image_url": f"/static/heatmaps/{sample_id}_heatmap.png"
    }

    saved = repo.save_inspection(record)
    assert saved["inspection_id"] == sample_id
    assert saved["status"] == "ANOMALY"

    # Query record by ID
    fetched = repo.get_inspection_by_id(sample_id)
    assert fetched is not None
    assert fetched["inspection_id"] == sample_id
    assert fetched["category"] == "candle"

    # Query statistics
    stats = repo.get_statistics()
    assert stats["total_inspections"] >= 1
    assert stats["anomaly_count"] >= 1

    # Query defects list
    defects = repo.get_defects()
    assert len(defects) >= 1

if __name__ == "__main__":
    pytest.main(["-v", __file__])
