import sys

from huggingface_hub import login
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
control_img = args.validation_image or f"controlnet_dataset/images/sample_0000.jpg"
training_command = [
    "accelerate", "launch", training_script,
    "--pretrained_model_name_or_path", base_model,
    "--controlnet_model_name_or_path", controlnet_model,
    "--output_dir", output_dir,
    "--conditioning_image_column", "condition_image",
    "--image_column", "image",
    "--caption_column", "prompt",
    f"--jsonl_for_train", f"../controlnet_dataset/dataset_{args.controlnet_type.lower()}.jsonl",
    "--resolution", args.resolution,
    "--learning_rate", args.learning_rate,
    "--max_train_steps", args.steps,
    "--checkpointing_steps", args.checkpointing_steps,
    "--validation_steps", args.validation_steps,
    "--mixed_precision", args.mixed_precision,
    "--validation_image", control_img,
    "--validation_prompt", args.prompt,
    "--train_batch_size", args.train_batch_size,
    "--gradient_accumulation_steps", args.gradient_accumulation_steps,
    "--gradient_checkpointing",
    "--use_8bit_adam",
    "--set_grads_to_none",
    "--push_to_hub",
    "--controlnet_type", args.controlnet_type.lower(),
    "--N4", args.N4,
    "--hub_model_id", args.hub_model_id
]

print("Esecuzione comando Accelerate:")
print(" ".join(map(str, training_command)), flush=True)

process = subprocess.Popen(
    list(map(str, training_command)),
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

print("\n[TRAINING_COMPLETE]\n", flush=True)