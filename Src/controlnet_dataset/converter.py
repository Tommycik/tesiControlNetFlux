from controlnet_aux import CannyDetector
from PIL import Image
import os
import cv2
import numpy as np

# Import the run_hed function
from imagesToHed import run_hed

# Paths
input_dir = "images"
canny_dir = "imagesControlCanny"
hed_dir = "imagesControlHed"

os.makedirs(canny_dir, exist_ok=True)
os.makedirs(hed_dir, exist_ok=True)

canny = CannyDetector()

def load_image_safe(filepath):
    pil_img = Image.open(filepath)
    if pil_img.mode == "RGBA":
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img

for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    if not os.path.isfile(filepath):
        continue
    name, ext = os.path.splitext(filename)

    try:
        pil_img = load_image_safe(filepath)
    except Exception as e:
        print(f"Skipping {filename}, invalid image: {e}")
        continue

    # Canny
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(os.path.join(canny_dir, f"{name}_canny.png"), edges)

    # HED
    hed_out = os.path.join(hed_dir, f"{name}_hed.png")
    run_hed(filepath, hed_out)

print("Done.")
