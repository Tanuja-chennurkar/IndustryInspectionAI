"""
Automated PyYAML Syntax Validator for Kubernetes Manifest Files.
Parses all 11 Kubernetes manifests and validates standard API fields.
"""

import os
import glob
import yaml
import pytest

K8S_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../k8s"))

def test_k8s_manifests_syntax():
    yaml_files = glob.glob(os.path.join(K8S_DIR, "**/*.yaml"), recursive=True)
    assert len(yaml_files) >= 11, f"Expected at least 11 k8s manifests, found {len(yaml_files)}"

    for yfile in yaml_files:
        with open(yfile, "r") as f:
            docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if not doc:
                    continue
                assert "apiVersion" in doc, f"Missing apiVersion in {yfile}"
                assert "kind" in doc, f"Missing kind in {yfile}"
                assert "metadata" in doc, f"Missing metadata in {yfile}"
                assert "name" in doc["metadata"], f"Missing metadata.name in {yfile}"
                print(f"[k8s Validated] Kind: {doc['kind']:<15} Name: {doc['metadata']['name']}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
