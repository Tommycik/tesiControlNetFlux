import os
import csv
import json
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch

IMAGES_DIR = "images"
CAPTIONS_FILE = "New_Captition.txt"
OUT_BASE = "controlnet_dataset"
IMAGES_HED_DIR = OUT_BASE + "/"+"imagesControlHed"
IMAGES_CANNY_DIR = OUT_BASE + "/"+"imagesControlCanny"

def generate_dataset():
    # Read captions file robustly
    df = pd.read_csv(
        CAPTIONS_FILE,
        header=None,
        names=["orig", "caption"],
        quoting=csv.QUOTE_MINIMAL,
        dtype=str,
        engine='python'
    )
    df = df.dropna(subset=["orig", "caption"]).reset_index(drop=True)

    dataset_canny = []
    dataset_hed = []

    for idx, row in df.iterrows():
        orig = str(row["orig"]).strip()
        caption = str(row["caption"]).strip()

        # source path
        src = os.path.join(IMAGES_DIR, orig)
        if not os.path.exists(src):
            # try case-insensitive search
            found = None
            parent = Path(IMAGES_DIR)
            lower_target = orig.lower()
            for p in parent.glob("**/*"):
                if p.is_file() and p.name.lower() == lower_target:
                    found = str(p)
                    break
            if found is not None:
                src = found
                print(f"Found case-insensitive match for {orig} -> {src}")
            else:
                print(f"Warning: image not found: {src}. Skipping all captions for this filename.")
                continue

        # keep original filename
        sample_name = OUT_BASE+"/"+IMAGES_DIR + "/" + f"{Path(orig).stem}.png"

        # JSONL rows: point to existing processed control images
        canny_path = IMAGES_CANNY_DIR + "/" + f"{Path(orig).stem}_canny.png"
        hed_path = IMAGES_HED_DIR + "/" + f"{Path(orig).stem}_hed.jpg"

        dataset_canny.append({
            "image": sample_name,
            "condition_image": canny_path,
            "prompt": caption
        })
        dataset_hed.append({
            "image": sample_name,
            "condition_image": hed_path,
            "prompt": caption
        })

    # Save JSONL files
    with open("dataset_canny.jsonl", "w", encoding="utf-8") as f:
        for row in dataset_canny:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open("dataset_hed.jsonl", "w", encoding="utf-8") as f:
        for row in dataset_hed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Dataset JSONL generated (keeping original image names).")

if __name__ == "__main__":
    generate_dataset()