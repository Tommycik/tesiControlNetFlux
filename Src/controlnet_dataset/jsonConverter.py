import json
import os

with open("dataset.json", "r") as f:
    data = json.load(f)

for item in data:
    if "image" in item:
        item["image"] = os.path.join("./controlnet_dataset", item["image"])
    if "condition_image" in item:
        item["condition_image"] = os.path.join("./controlnet_dataset", item["condition_image"])

with open("dataset.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")