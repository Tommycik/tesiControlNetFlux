import os
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
# Input dir
images_dir = Path("content/image")
canny_dir = Path("content/canny")
prompts_file = Path("content/captions_eng.csv")

# Output dir
output_dir = Path("")
output_dir.mkdir(parents=True, exist_ok=True)

# Leggi i prompt
with open(prompts_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines()]

image_files = sorted(images_dir.glob("*"))
canny_files = sorted(canny_dir.glob("*"))

assert len(image_files) == len(canny_files)

dataset = []

#todo modificare per nuovo dataset
print("Creating dataset")
for idx, (img_path, canny_path, prompt) in tqdm(enumerate(zip(image_files, canny_files, prompts)), total=len(prompts)):
    base_name = f"sample_{idx:04d}"

    # Copia o referenzia i file nel dataset
    new_img_path = output_dir / f"images/{base_name}.jpg"
    new_canny_path = output_dir / f"imagesControlCanny/{base_name}_canny.jpg"

    # Copy files
    os.system(f'copy  "{img_path}" "{new_img_path}"')
    os.system(f'copy  "{canny_path}" "{new_canny_path}"')

    dataset.append({
        "image": "controlnet_dataset/"+new_img_path.name,
        "condition_image": "controlnet_dataset/"+new_canny_path.name,
        "prompt": prompt
    })

# Saving json
with open(output_dir / "dataset_canny.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)



input_dir = 'images'
output_dir = 'imagesControlHed'
model = 'bsds500'  # You can change this if needed

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all image files from the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_hed.jpg')

        # Run the HED script using subprocess
        subprocess.run([
            'python', 'imagesToHed.py',
            '--model=' + model,
            '--in=' + input_path,
            '--out=' + output_path
        ])
# Load the existing JSON
with open("dataset_canny.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace "imagesControlCanny/XYZ_canny.jpg" with "imagesControlHed/XYZ_hed.jpg"
for entry in data:
    old_path = entry["condition_image"]
    filename = old_path.split("/")[-1].replace("_canny", "_hed")
    entry["condition_image"] = f"imagesControlHed/{filename}"

# Save the modified JSON
with open("dataset_hed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

with open("dataset_canny.json", "r") as f:
    data = json.load(f)

with open("dataset_canny.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

with open("dataset_hed.json", "r") as f:
    data = json.load(f)

with open("dataset_hed.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
print(f"✅ Dataset creato in: {output_dir}")