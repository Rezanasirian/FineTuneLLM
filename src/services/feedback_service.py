import json
from pathlib import Path

def save_feedback(feedback: dict):
    path = Path('data/feedback') / f"{feedback['id']}.json"
    with open(path, 'w') as f:
        json.dump(feedback, f)