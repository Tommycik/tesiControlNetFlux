import json

#aggiorna la psozione dei file
# Percorso del file JSON
json_path = "dataset_canny.json"

# Carica il file JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Modifica ciascun elemento
for entry in data:
    image_name = entry["image"]
    condition_name = entry["condition_image"]

    # Aggiorna i path (es. aggiungi sottocartelle)
    entry["image"] = f"images/{image_name}"
    entry["condition_image"] = f"imagesControlCanny/{condition_name}"

    # (Opzionale) modifica il prompt
    # entry["prompt"] = "new prompt text here"

# Salva di nuovo il file
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("JSON modificato con successo.")