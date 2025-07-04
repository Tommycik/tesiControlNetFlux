import os
import json
from pathlib import Path
from tqdm import tqdm

# Cartelle input
images_dir = Path("content/image")
canny_dir = Path("content/canny")
prompts_file = Path("content/captions_eng.csv")

# Cartella output
output_dir = Path("controlnet_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

# Leggi i prompt
with open(prompts_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f.readlines()]

# Controlla che il numero di file sia coerente
image_files = sorted(images_dir.glob("*"))
canny_files = sorted(canny_dir.glob("*"))

assert len(image_files) == len(prompts) == len(canny_files), "Il numero di immagini, canny e prompt deve essere uguale!"

# Prepara dataset
dataset = []

print("Creazione dataset...")
for idx, (img_path, canny_path, prompt) in tqdm(enumerate(zip(image_files, canny_files, prompts)), total=len(prompts)):
    base_name = f"sample_{idx:04d}"

    # Copia o referenzia i file nel dataset
    new_img_path = output_dir / f"images/{base_name}.jpg"
    new_canny_path = output_dir / f"imagesControlCanny{base_name}_canny.jpg"

    # Copia i file (puoi anche fare symlink se preferisci)
    os.system(f'copy  "{img_path}" "{new_img_path}"')
    os.system(f'copy  "{canny_path}" "{new_canny_path}"')

    # Aggiungi al JSON dataset
    dataset.append({
        "image": new_img_path.name,
        "condition_image": new_canny_path.name,
        "prompt": prompt
    })

# Salva il file JSON con tutte le associazioni
with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"✅ Dataset creato in: {output_dir}")