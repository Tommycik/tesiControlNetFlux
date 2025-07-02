import cv2
import os

# Cartella contenente le immagini originali
input_folder = "images"
# Cartella dove salvare le immagini Canny (può essere la stessa)
output_folder = "imagesCanny"

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Estensioni supportate
supported_exts = [".jpg"]

# Scorri tutte le immagini nella cartella
for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() in supported_exts:
        input_path = os.path.join(input_folder, filename)

        # Legge e converte in scala di grigi
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applica Canny
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)

        # Costruisce il nuovo nome file
        output_filename = f"{name}__canny.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Salva il risultato
        cv2.imwrite(output_path, edges)
        print(f"Salvato: {output_path}")
