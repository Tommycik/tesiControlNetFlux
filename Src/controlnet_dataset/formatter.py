import os
import shutil
# mette le immagini nelle proprie cartelle
# Imposta la cartella di origine (modifica questo percorso se necessario)
source_folder = "./"  # Cartella corrente

# Cartelle di destinazione
images_folder = os.path.join(source_folder, "images")
controlimages_folder = os.path.join(source_folder, "imagesControl")

# Crea le cartelle di destinazione se non esistono
os.makedirs(images_folder, exist_ok=True)
os.makedirs(controlimages_folder, exist_ok=True)

# Scorri tutti i file nella cartella di origine
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") and not filename.endswith("_canny.jpg"):
        base_name = filename[:-4]  # Rimuove ".jpg"
        canny_filename = f"{base_name}_canny.jpg"

        source_image_path = os.path.join(source_folder, filename)
        source_canny_path = os.path.join(source_folder, canny_filename)

        # Verifica se il file _canny esiste
        if os.path.exists(source_canny_path):
            # Sposta l'immagine originale
            shutil.move(source_image_path, os.path.join(images_folder, filename))
            # Sposta l'immagine canny
            shutil.move(source_canny_path, os.path.join(controlimages_folder, canny_filename))
            print(f"Moved: {filename} -> images/, {canny_filename} -> imagesControl/")