import os
import sys
import cv2
import pytest
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.inference.detector import AnomalyDetector

def test_manual_category_routing():
    """
    Task 4 Diagnostic Test:
    Verify that explicit category routing (bypassing automatic classifier)
    correctly evaluates anomaly scores and returns status.
    """
    detector = AnomalyDetector(saved_models_dir="ml/saved_models")

    capsules_normal_path = os.path.abspath("test_images/capsules_normal.jpg")
    capsules_defective_path = os.path.abspath("test_images/capsules_defective.jpg")

    assert os.path.exists(capsules_normal_path), "test_images/capsules_normal.jpg missing"
    assert os.path.exists(capsules_defective_path), "test_images/capsules_defective.jpg missing"

    # Test 1: Manual routing for normal capsule sample
    res_norm = detector.inspect_image(
        image_input=capsules_normal_path,
        category="capsules"
    )
    print("\n--- Manual Routing: Capsules Normal ---")
    print(f"Category: {res_norm['category']}")
    print(f"Status: {res_norm['status']}")
    print(f"Score: {res_norm['anomaly_score']} | Threshold: {res_norm['threshold']}")
    print(f"Defect Area: {res_norm['defect_area_percentage']}%")

    assert res_norm["category"] == "capsules"
    assert res_norm["status"] == "NORMAL"
    assert res_norm["defect_area_percentage"] == 0.0

    # Test 2: Manual routing for defective capsule sample
    res_def = detector.inspect_image(
        image_input=capsules_defective_path,
        category="capsules"
    )
    print("\n--- Manual Routing: Capsules Defective ---")
    print(f"Category: {res_def['category']}")
    print(f"Status: {res_def['status']}")
    print(f"Score: {res_def['anomaly_score']} | Threshold: {res_def['threshold']}")
    print(f"Defect Area: {res_def['defect_area_percentage']}%")

    assert res_def["category"] == "capsules"
    assert res_def["status"] == "ANOMALY"
    assert res_def["anomaly_score"] >= res_def["threshold"]
    assert res_def["defect_area_percentage"] > 0.0

    print("\n[SUCCESS] Task 4 Manual Routing Diagnostic Test Passed!")

if __name__ == "__main__":
    test_manual_category_routing()
