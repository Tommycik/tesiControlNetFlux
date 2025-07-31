# controlnet_infer_api.py
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
parser.add_argument('--N4', type=bool,required=True, default=False)
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

control_img = f"controlnet_dataset/imagesControlHed/sample_0000_{args.controlnet_type}.jpg"
control_image = load_image(control_img)

result = pipe(
    args.prompt,
    control_image=control_image,
    controlnet_conditioning_scale=args.scale,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
).images[0]

# Upload to Cloudinary
img_byte_arr = BytesIO()
result.save(img_byte_arr, format='JPEG')
img_byte_arr.seek(0)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
public_id = f"flux_controlnet_hed_results/{timestamp}_{uuid.uuid4().hex[:8]}"
response = cloudinary.uploader.upload(img_byte_arr, public_id=public_id,folder=args.args.controlnet_model,
            resource_type="image")
print(response["secure_url"])