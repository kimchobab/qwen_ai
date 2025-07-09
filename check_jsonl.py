import json

file_path = "docs/data/train.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            json.loads(line)
        except Exception as e:
            print(f" Line {i}: {e}")
