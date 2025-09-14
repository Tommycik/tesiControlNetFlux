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
import shutil

# -------------------------
# Config
# -------------------------
IMAGES_DIR = "images"                               # where your original images live
CAPTIONS_FILE = "New_Captition.txt"                 # CSV-like (filename, "caption")
OUT_BASE = "controlnet_dataset"
IMAGES_OUT = os.path.join(OUT_BASE, "images")
CANNY_OUT = os.path.join(OUT_BASE, "imagesControlCanny")
HED_OUT = os.path.join(OUT_BASE, "imagesControlHed")

os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(CANNY_OUT, exist_ok=True)
os.makedirs(HED_OUT, exist_ok=True)

# HED model global holder
_net_hed = None

# -------------------------
# Utility: robust PIL loader (drops alpha, strips bad ICC)
# -------------------------
def load_image_rgb(path: str) -> Image.Image:
    """
    Open with PIL, convert to RGB (drops alpha/palette), return PIL Image.
    """
    img = Image.open(path)
    img = img.convert("RGB")
    return img

# -------------------------
# Canny: clean pipeline
# -------------------------
def auto_canny_edges(pil_img: Image.Image, sigma: float = 0.33, blur_ks=(5,5), blur_sigma=1.0):
    """
    Returns a uint8 HxW numpy array with edges (0..255).
    Uses RGB->GRAY, gaussian blur, median-based automatic thresholds.
    """
    arr = np.array(pil_img)  # shape (H,W,3), uint8
    # Ensure type uint8
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # small blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, blur_ks, blur_sigma)

    median = np.median(gray_blur)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    # safety fallback in case lower >= upper
    if lower >= upper:
        lower, upper = 50, 150

    # Use L2gradient for slightly better edge quality
    edges = cv2.Canny(gray_blur, lower, upper, L2gradient=True)
    return edges  # uint8 HxW

# -------------------------
# HED: small implementation using network-bsds500.pytorch
# (based on the original public code by S. Niklaus)
# -------------------------
class _HEDNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU(False)
        )
        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU(False)
        )
        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(False)
        )
        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(False)
        )
        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(False)
        )

        self.netScoreOne = torch.nn.Conv2d(64, 1, 1, 1, 0)
        self.netScoreTwo = torch.nn.Conv2d(128, 1, 1, 1, 0)
        self.netScoreThr = torch.nn.Conv2d(256, 1, 1, 1, 0)
        self.netScoreFou = torch.nn.Conv2d(512, 1, 1, 1, 0)
        self.netScoreFiv = torch.nn.Conv2d(512, 1, 1, 1, 0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(5, 1, 1, 1, 0), torch.nn.Sigmoid()
        )

    def forward(self, ten_input):
        # ten_input expected [B,3,H,W] with values in [0,1]
        ten_input = ten_input * 255.0
        mean = torch.tensor([104.00698793, 116.66876762, 122.67891434], dtype=ten_input.dtype, device=ten_input.device).view(1,3,1,1)
        ten_input = ten_input - mean

        ten_vgg_one = self.netVggOne(ten_input)
        ten_vgg_two = self.netVggTwo(ten_vgg_one)
        ten_vgg_thr = self.netVggThr(ten_vgg_two)
        ten_vgg_fou = self.netVggFou(ten_vgg_thr)
        ten_vgg_fiv = self.netVggFiv(ten_vgg_fou)

        s1 = self.netScoreOne(ten_vgg_one)
        s2 = self.netScoreTwo(ten_vgg_two)
        s3 = self.netScoreThr(ten_vgg_thr)
        s4 = self.netScoreFou(ten_vgg_fou)
        s5 = self.netScoreFiv(ten_vgg_fiv)

        s1 = torch.nn.functional.interpolate(s1, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        s2 = torch.nn.functional.interpolate(s2, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        s3 = torch.nn.functional.interpolate(s3, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        s4 = torch.nn.functional.interpolate(s4, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)
        s5 = torch.nn.functional.interpolate(s5, size=(ten_input.shape[2], ten_input.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([s1,s2,s3,s4,s5], 1))

def _ensure_hed_loaded(cache_dir=None):
    global _net_hed
    if _net_hed is not None:
        return _net_hed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _HEDNet().to(device)

    # download original bsds500 weights (hosted by sniklaus)
    url = "http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch"
    try:
        state = torch.hub.load_state_dict_from_url(url, progress=True)
    except Exception as e:
        raise RuntimeError(f"Failed to download HED weights from {url}: {e}")

    # some checkpoints were saved under module.* keys
    new_state = {}
    for k, v in state.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        # `net` prefix may be expected by older versions; we attempt to map common patterns:
        new_state[new_k] = v

    net.load_state_dict(new_state, strict=False)
    net.eval()
    _net_hed = net
    return _net_hed

def run_hed_on_pil(pil_img: Image.Image, device=None):
    """
    Run HED on a PIL RGB image and return a uint8 HxW numpy array.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _ensure_hed_loaded()

    arr = np.array(pil_img)  # H,W,3 uint8
    # BGR ordering expected by original net (we will swap below)
    # convert to float [0,1]
    arr_f = arr.astype(np.float32) / 255.0
    # convert to BGR for this implementation, because original used BGR mean subtraction
    arr_bgr = arr_f[:, :, ::-1]
    ten = torch.from_numpy(np.ascontiguousarray(arr_bgr.transpose(2,0,1))).unsqueeze(0).to(device).float()
    with torch.no_grad():
        out = net(ten)[0, 0].cpu().numpy()  # shape HxW
    # normalize to 0..255
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    out_u8 = (out * 255.0).clip(0,255).astype(np.uint8)
    return out_u8

# -------------------------
# Main pipeline
# -------------------------
def generate_dataset():
    # Read captions file robustly: file may contain lines like
    # filename.png,"caption, possibly with commas"
    df = pd.read_csv(CAPTIONS_FILE, header=None, names=["orig", "caption"], quoting=csv.QUOTE_MINIMAL, dtype=str, engine='python')
    # Some rows may be NaN; drop them
    df = df.dropna(subset=["orig", "caption"]).reset_index(drop=True)

    mapping = {}   # orig filename -> sample_xxxx.png
    next_idx = 0

    dataset_canny = []
    dataset_hed = []

    for idx, row in df.iterrows():
        orig = str(row["orig"]).strip()
        caption = str(row["caption"]).strip()

        # assign sample name on first encounter
        if orig not in mapping:
            sample_name = f"sample_{next_idx:04d}.png"
            mapping[orig] = sample_name
            next_idx += 1

            # source path
            src = os.path.join(IMAGES_DIR, orig)
            if not os.path.exists(src):
                # try a few common alternatives (case-insensitive) if missing
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
                    print(f"⚠️ Warning: image not found: {src}. Skipping all captions for this filename.")
                    # skip mapping so future captions referencing same name will be skipped
                    mapping[orig] = None
                    continue

            try:
                # load + normalize
                pil = load_image_rgb(src)

                # copy / rename canonical original to dataset images (strip problematic ICC)
                dst_original = os.path.join(IMAGES_OUT, sample_name)
                shutil.copy2(src, dst_original)

                # produce canny
                edges = auto_canny_edges(pil)
                canny_name = sample_name.replace(".png", "_canny.png")
                canny_path = os.path.join(CANNY_OUT, canny_name)
                cv2.imwrite(canny_path, edges)
                # produce hed
                try:
                    hed_arr = run_hed_on_pil(pil)
                    hed_name = sample_name.replace(".png", "_hed.png")
                    hed_path = os.path.join(HED_OUT, hed_name)
                    cv2.imwrite(hed_path, hed_arr)
                except Exception as e:
                    print(f"⚠️ HED failed for {src}: {e}. Saving fallback Sobel as hed.")
                    # fallback: simple Sobel magnitude
                    arr = np.array(pil)
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    mag = np.sqrt(sx*sx + sy*sy)
                    mag = (255.0 * (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)).astype(np.uint8)
                    hed_name = sample_name.replace(".png", "_hed.png")
                    hed_path = os.path.join(HED_OUT, hed_name)
                    Image.fromarray(mag, mode="L").save(hed_path)

            except Exception as e:
                print(f"ERROR processing {src}: {e}")
                mapping[orig] = None
                continue

        # now construct the JSONL rows (skip if mapping ended up None)
        sample = mapping[orig]
        if sample is None:
            continue

        dataset_canny.append({
            "image": f"{OUT_BASE}/images/{sample}",
            "condition_image": f"{OUT_BASE}/imagesControlCanny/{sample.replace('.png','_canny.png')}",
            "text": caption
        })
        dataset_hed.append({
            "image": f"{OUT_BASE}/images/{sample}",
            "condition_image": f"{OUT_BASE}/imagesControlHed/{sample.replace('.png','_hed.png')}",
            "text": caption
        })

    # write jsonl
    with open("dataset_canny.jsonl", "w", encoding="utf-8") as f:
        for r in dataset_canny:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("dataset_hed.jsonl", "w", encoding="utf-8") as f:
        for r in dataset_hed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Done. Samples processed:", next_idx)
    print("Wrote dataset_canny.jsonl and dataset_hed.jsonl")

if __name__ == "__main__":
    # optionally silence the libpng iCCP warning if you still see it
    warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")
    generate_dataset()