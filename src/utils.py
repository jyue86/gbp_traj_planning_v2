import json
from typing import Dict

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)