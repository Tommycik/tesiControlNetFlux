import sys
import tempfile

import yaml
from huggingface_hub import HfApi, login
from datasets import load_dataset
import argparse
import os
import subprocess
import torch
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='transparent glass on white background, the bottom part of the glass presents light grooves')
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--resolution', type=int, default=512)
parser.add_argument('--checkpointing_steps', type=int, default=250)
parser.add_argument('--validation_steps', type=int, default=125)
parser.add_argument('--mixed_precision', type=str, default='bf16')
parser.add_argument('--controlnet_model', type=str,required=True)
parser.add_argument('--controlnet_type', type=str,required=True)
parser.add_argument('--learning_rate', type=str, default='2e-6')
parser.add_argument('--N4', type=bool, default=False)
parser.add_argument('--hub_model_id', required=True, type=str)
parser.add_argument('--validation_image', type=str, default=None)
args = parser.parse_args()

login(token=os.environ["HUGGINGFACE_TOKEN"])

output_dir = "model"
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = args.controlnet_model
training_script = "scripts/train_controlnet_flux.py"
from pathlib import Path

# Validation image
validation_image_path = args.validation_image or "../controlnet_dataset/images/sample_0000.jpg"
validation_image_path = Path(__file__).resolve().parent / validation_image_path
validation_image_path = validation_image_path.resolve()
if not validation_image_path.is_file():
    raise FileNotFoundError(f"Validation image not found at {validation_image_path}")

# JSON dataset
jsonl_path = f"../controlnet_dataset/dataset_{args.controlnet_type.lower()}.jsonl"
jsonl_path = Path(__file__).resolve().parent / jsonl_path
jsonl_path = jsonl_path.resolve()
if not jsonl_path.is_file():
    raise FileNotFoundError(f"Dataset JSON not found at {jsonl_path}")
training_command = [
    "accelerate", "launch", training_script,
    "--pretrained_model_name_or_path", base_model,
    "--controlnet_model_name_or_path", controlnet_model,
    "--output_dir", output_dir,
    "--conditioning_image_column", "condition_image",
    "--image_column", "image",
    "--caption_column", "prompt",
    "--jsonl_for_train", str(jsonl_path),
    "--resolution", str(args.resolution),
    "--learning_rate", str(args.learning_rate),
    "--max_train_steps", str(args.steps),
    "--checkpointing_steps", str(args.checkpointing_steps),
    "--validation_steps", str(args.validation_steps),
    "--mixed_precision", str(args.mixed_precision),
    "--validation_image", str(validation_image_path),
    "--validation_prompt", str(args.prompt),
    "--train_batch_size", str(args.train_batch_size),
    "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
    "--controlnet_type", str(args.controlnet_type.lower()),
    "--N4", str(args.N4).capitalize(),
    "--hub_model_id", str(args.hub_model_id),

]

print("Esecuzione comando Accelerate:")
print(" ".join(map(str, training_command)), flush=True)

process = subprocess.Popen(
    training_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in iter(process.stdout.readline, ''):
    print(line, flush=True)  # flush so logs reach Flask/SSH

ret = process.wait()
if ret != 0:
    print(f"[ERROR] Training failed with exit code {ret}", flush=True)
    sys.exit(ret)

train_config = {
            "controlnet_type": args.controlnet_type,
            "controlnet_model": args.controlnet_model,
            "N4": args.N4,
            "mixed_precision": args.mixed_precision,
            "steps": args.steps,
            "train_batch_size": args.train_batch_size,
            "learning_rate": args.learning_rate,
            "resolution": args.resolution,
            "checkpointing_steps": args.checkpointing_steps,
            "validation_steps": args.validation_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "validation_image": args.validation_image_path or "default",
            "hub_model_id": args.hub_model_id,
        }

yaml_path = os.path.join(tempfile.gettempdir(), "training_config.yaml")
with open(yaml_path, "w") as f:
    yaml.safe_dump(train_config, f)

api = HfApi()

# Now you can upload
api.upload_file(
    path_or_fileobj=yaml_path,
    path_in_repo="training_config.yaml",
    repo_id=args.hub_model_id,
    repo_type="model",
    token=os.environ["HUGGINGFACE_TOKEN"]
)
print("\n[TRAINING_COMPLETE]\n", flush=True)