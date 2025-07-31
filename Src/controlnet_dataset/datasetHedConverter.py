import json

# Load the existing JSON
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace "imagesControlCanny/XYZ_canny.jpg" with "imagesControlHed/XYZ_hed.jpg"
for entry in data:
    old_path = entry["condition_image"]
    filename = old_path.split("/")[-1].replace("_canny", "_hed")
    entry["condition_image"] = f"imagesControlHed/{filename}"

# Save the modified JSON
with open("dataset_hed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)