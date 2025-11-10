from pathlib import Path
import json

def save_json(data, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
