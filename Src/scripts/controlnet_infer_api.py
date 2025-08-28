import argparse
import os

from huggingface_hub import login
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from io import BytesIO
from datetime import datetime
from transformers import BitsAndBytesConfig
import cloudinary
import cloudinary.uploader
import torch
import uuid

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--scale', type=float, default=0.2)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--guidance', type=float, default=6.0)
parser.add_argument('--controlnet_model', type=str,required=True, default='tommycik/controlFluxAlcol')
parser.add_argument('--controlnet_type', type=str,required=True, default='canny')
parser.add_argument('--N4', type=bool, required=True, default=False)
parser.add_argument('--control_image', type=str, default=None)
args = parser.parse_args()

login(token=os.environ["HUGGINGFACE_TOKEN"])

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = args.controlnet_model

if(args.N4):
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
from pathlib import Path

control_img = args.control_image or f"./controlnet_dataset/imagesControl{controlnet_type_capitalized}/sample_0000_{args.controlnet_type}.jpg"
control_image = load_image(control_img)

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

# Carica immagine generata
img_byte_arr = BytesIO()
result.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)
response = cloudinary.uploader.upload(
    img_byte_arr,
    public_id=base_public_id,
    folder="repo_image",
    resource_type="image"
)
print(response["secure_url"])

# Carica immagine di controllo
control_img_byte_arr = BytesIO()
control_image.save(control_img_byte_arr, format='JPEG')
control_img_byte_arr.seek(0)
response_control = cloudinary.uploader.upload(
    control_img_byte_arr,
    public_id=f"{base_public_id}_control",
    folder="repo_control",
    resource_type="image"
)

# Carica file di testo con prompt e parametri
text_content = f"Prompt: {args.prompt}\nScale: {args.scale}\nSteps: {args.steps}\nGuidance: {args.guidance}\nControlnet_model: {args.controlnet_model}\nControlnet_type: {args.controlnet_type}\nN4: {args.N4}\n"
text_byte_arr = BytesIO(text_content.encode('utf-8'))
response_text = cloudinary.uploader.upload(
    text_byte_arr,
    public_id=f"{base_public_id}_text",
    folder="repo_text",
    resource_type="raw"
)