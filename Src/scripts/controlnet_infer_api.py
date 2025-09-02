import argparse
import os

from huggingface_hub import login, HfApi
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
parser.add_argument('--N4', action='store_true')
parser.add_argument('--control_image', type=str, default=None)
args = parser.parse_args()

api = HfApi()
def validate_model_or_fallback(model_id: str, default_model: str):
    """Controlla se il repo contiene un config.json, altrimenti torna al default."""
    try:
        files = api.list_repo_files(model_id)
        if "config.json" in files:
            return model_id
        else:
            print(f"[WARNING] Repo {model_id} non valido, uso fallback {default_model}")
            return default_model
    except Exception as e:
        print(f"[ERROR] Impossibile accedere al repo {model_id}: {e}")
        return default_model

login(token=os.environ["HUGGINGFACE_TOKEN"])

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = args.controlnet_model
default_canny = "InstantX/FLUX.1-dev-Controlnet-Canny"
controlnet_model = validate_model_or_fallback(controlnet_model, default_canny)

if args.N4:
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model,
        quantization_config=bnb_config,
        # device_map="auto" not supported
    )
    pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)

else:

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)

pipe.to("cuda")

controlnet_type_capitalized = args.controlnet_type.capitalize()
from pathlib import Path

control_img = args.control_image or f"controlnet_dataset/imagesControl{controlnet_type_capitalized}/sample_0000_{args.controlnet_type}.jpg"
control_img_path = Path(__file__).resolve().parent.parent / control_img
control_img_path = control_img_path.resolve()

if not control_img_path.is_file():
    raise FileNotFoundError(f"File not found: {control_img_path}")

control_image = load_image(str(control_img_path))

def progress_callback(step: int, timestep: int, latents):
    print(f"[PROGRESS] {step}/{args.steps}", flush=True)

pipe.scheduler.set_timesteps(args.steps)
pipe.enable_attention_slicing()  # optional, saves VRAM

# Prepare latents
latents = pipe.prepare_latents(batch_size=1, image=control_image, generator=None)

for step, t in enumerate(pipe.scheduler.timesteps, 1):
    # Predict noise
    noise_pred = pipe.unet(
        latents,
        t,
        encoder_hidden_states=pipe._encode_prompt(args.prompt, device="cuda")
    )[0]

    # Classifier-free guidance
    if args.guidance > 1.0:
        noise_pred_uncond = pipe.unet(
            latents,
            t,
            encoder_hidden_states=pipe._encode_prompt("", device="cuda")
        )[0]
        noise_pred = noise_pred_uncond + args.guidance * (noise_pred - noise_pred_uncond)

    # Step scheduler
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Print real-time progress for API caller
    print(f"[PROGRESS] Step {step}/{args.steps}", flush=True)

# Decode final image
result = pipe.decode_latents(latents)[0]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
unique_id = uuid.uuid4().hex[:8]
base_public_id = f"{timestamp}_{unique_id}"
folder_base = f"{args.controlnet_model}_results"
folder_image = f"{folder_base}/repo_image"
folder_control = f"{folder_base}/repo_control"
folder_text = f"{folder_base}/repo_text"
# Carica immagine generata
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

# Carica immagine di controllo
control_img_byte_arr = BytesIO()
control_image.save(control_img_byte_arr, format='JPEG')
control_img_byte_arr.seek(0)
response_control = cloudinary.uploader.upload(
    control_img_byte_arr,
    public_id=f"{base_public_id}_control",
    folder=folder_control,
    resource_type="image"
)

# Carica file di testo con prompt e parametri
text_content = f"Prompt: {args.prompt}\nScale: {args.scale}\nSteps: {args.steps}\nGuidance: {args.guidance}\nControlnet_model: {args.controlnet_model}\nControlnet_type: {args.controlnet_type}\nN4: {args.N4}\n"
text_byte_arr = BytesIO(text_content.encode('utf-8'))
response_text = cloudinary.uploader.upload(
    text_byte_arr,
    public_id=f"{base_public_id}_text",
    folder=folder_text,
    resource_type="raw"
)