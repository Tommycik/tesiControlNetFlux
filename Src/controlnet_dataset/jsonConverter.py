import json


with open("dataset.json", "r") as f:
    data = json.load(f)


with open("dataset.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
