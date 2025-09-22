import argparse
import os
import sys
import cloudinary
import cloudinary.uploader
import torch
import uuid
import logging
import builtins

from huggingface_hub import login, HfApi
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from io import BytesIO
from datetime import datetime
from transformers import BitsAndBytesConfig
from diffusers import logging as diffusers_logging
from tqdm import tqdm as original_tqdm
from pathlib import Path

def line_tqdm(*args, **kwargs):
    kwargs.update(dict(
        mininterval=0,
        miniters=1,
        leave=True,
        file=sys.stdout,
        dynamic_ncols=False
    ))
    # critical: disable carriage return
    t = original_tqdm(*args, **kwargs)
    t.display = lambda msg=None, pos=None: sys.stdout.write((msg or t.__str__()) + "\n") or sys.stdout.flush()
    return t


builtins.tqdm = line_tqdm
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

diffusers_logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--scale', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--guidance', type=float, default=6.0)
parser.add_argument('--controlnet_model', type=str,required=True, default='tommycik/controlFluxAlcol')
parser.add_argument('--controlnet_type', type=str,required=True, default='canny')
parser.add_argument('--N4', action='store_true')
parser.add_argument('--control_image', type=str, default=None)
args = parser.parse_args()

api = HfApi()
def validate_model_or_fallback(model_id: str, default_model: str):
    #check if model is valid
    try:
        files = api.list_repo_files(model_id)
        if "config.json" in files:
            return model_id
        else:
            print(f"[WARNING] Repository {model_id} not valid, using fallback {default_model}")
            return default_model
    except Exception as e:
        print(f"[ERROR] Repository {model_id}: {e} isn't accessible, using fallback {default_model}")
        return default_model

login(token=os.environ["HUGGINGFACE_TOKEN"])

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = args.controlnet_model
default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"
controlnet_model = validate_model_or_fallback(controlnet_model, default_canny)

if args.N4:
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model,
        quantization_config=bnb_config,
        # device_map="auto" not supported
    )
    pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)

else:

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)

pipe.to("cuda")

controlnet_type_capitalized = args.controlnet_type.capitalize()
type = args.controlnet_type.lower()
image_default = None
if type == "hed":
    image_default = "boccale_hed.jpg"
elif type == "canny":
    image_default = "boccale_canny.png"
else:
    image_default = "boccale_canny.png"


control_img = args.control_image or f"controlnet_dataset/controlImagesDefault/{image_default}"
control_img_path = Path(__file__).resolve().parent.parent / control_img
control_img_path = control_img_path.resolve()

if not control_img_path.is_file():
    raise FileNotFoundError(f"File not found: {control_img_path}")

control_image = load_image(str(control_img_path))

result = pipe(
    args.prompt,
    control_image=control_image,
    controlnet_conditioning_scale=args.scale,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
).images[0]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
unique_id = uuid.uuid4().hex[:8]
base_public_id = f"{timestamp}_{unique_id}"
folder_base = f"{args.controlnet_model}_results"
folder_image = f"{folder_base}/repo_image"
folder_control = f"{folder_base}/repo_control"
folder_text = f"{folder_base}/repo_text"
# Uploads generated image
img_byte_arr = BytesIO()
result.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)
response = cloudinary.uploader.upload(
    img_byte_arr,
    public_id=base_public_id,
    folder=folder_image,
    resource_type="image"
)
print(response["secure_url"], flush=True)

# Uploads control image
control_img_byte_arr = BytesIO()
control_image.save(control_img_byte_arr, format='JPEG')
control_img_byte_arr.seek(0)
response_control = cloudinary.uploader.upload(
    control_img_byte_arr,
    public_id=f"{base_public_id}_control",
    folder=folder_control,
    resource_type="image"
)

# Uploads parameters and prompt as text file
text_content = f"Prompt: {args.prompt}\nScale: {args.scale}\nSteps: {args.steps}\nGuidance: {args.guidance}\nControlnet_model: {args.controlnet_model}\nControlnet_type: {args.controlnet_type}\nN4: {args.N4}\n"
text_byte_arr = BytesIO(text_content.encode('utf-8'))
response_text = cloudinary.uploader.upload(
    text_byte_arr,
    public_id=f"{base_public_id}_text",
    folder=folder_text,
    resource_type="raw"
)