import json

with open("docs/data/train_flat.jsonl", "r", encoding="utf-8") as f:
    for _ in range(2):
        line = f.readline()
        print(json.loads(line))
