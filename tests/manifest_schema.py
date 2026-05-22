# tests/test_manifest.py
import json
import pytest
from pathlib import Path

def test_manifest_exists_and_is_valid():
    manifest_path = Path("artifacts/manifest.json")
    
    # Assert file exists
    assert manifest_path.exists(), "manifest.json not found in artifacts/"
    
    # Assert it is valid JSON
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    # Check for required keys (update these to match your actual manifest spec)
    required_keys = ["version", "timestamp", "feature_hash"]
    for key in required_keys:
        assert key in data, f"Manifest missing required key: {key}"