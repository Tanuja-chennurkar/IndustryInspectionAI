import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.inference.detector import AnomalyDetector

def test_reconstruction_quality():
    detector = AnomalyDetector(saved_models_dir="ml/saved_models")
    capsules_normal_path = os.path.abspath("test_images/capsules_normal.jpg")

    result = detector.inspect_image(image_input=capsules_normal_path, category="capsules")
    recon_rgb = result["reconstructed_image"]

    print("\n--- Task 7 Reconstruction Stats ---")
    print(f"Reconstructed Image shape: {recon_rgb.shape}")
    print(f"Reconstructed Image dtype: {recon_rgb.dtype}")
    print(f"Min intensity: {np.min(recon_rgb)}")
    print(f"Max intensity: {np.max(recon_rgb)}")
    print(f"Mean intensity: {np.mean(recon_rgb):.2f}")

    os.makedirs("scratch", exist_ok=True)
    out_path = os.path.abspath("scratch/test_reconstruction.png")
    cv2.imwrite(out_path, cv2.cvtColor(recon_rgb, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Reconstruction image saved to: {out_path}")

    assert recon_rgb.shape == (256, 256, 3)
    assert recon_rgb.dtype == np.uint8
    assert np.max(recon_rgb) > 50, "Reconstruction is nearly blank/gray!"
    print("[SUCCESS] Task 7 Reconstruction test passed!")

if __name__ == "__main__":
    test_reconstruction_quality()
