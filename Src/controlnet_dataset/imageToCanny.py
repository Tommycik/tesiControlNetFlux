import cv2
import os

# Input dir
input_folder = "images"
# Output dir
output_folder = "imagesControlCanny"

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Estensioni supportate
supported_exts = [".jpg"]

for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() in supported_exts:
        input_path = os.path.join(input_folder, filename)

        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        output_filename = f"{name}_canny.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Saving
        cv2.imwrite(output_path, edges)
        print(f"Salvato: {output_path}")
