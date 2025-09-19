import os
from PIL import Image


def check_image_channels(folder):
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            img = Image.open(filepath)
            print(f"{filename}: mode={img.mode}, channels={len(img.getbands())}")
        except Exception as e:
            print(f"Skipping {filename}, error: {e}")

if __name__ == "__main__":
    folder = "imagesControlHed"  # change this to your HED output directory
    check_image_channels(folder)