import pytest
import json
from pathlib import Path

def test_manifest_format():
    path = Path("artifacts/manifest.json")
    assert path.exists(), "Manifest file not found."
    with open(path, "r") as f:
        data = json.load(f)
    assert "version" in data, "Manifest missing version."
    # Add your specific schema assertions here