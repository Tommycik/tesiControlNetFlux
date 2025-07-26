import torch
import os
import uuid
import subprocess
from datetime import datetime
from huggingface_hub import login
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from transformers import BitsAndBytesConfig
import cloudinary
import cloudinary.uploader
from io import BytesIO # To save image to a buffer for Cloudinary upload

# --- Cloudinary Configuration ---
CLOUDINARY_CLOUD_NAME = "dz9gbl0lo"
CLOUDINARY_API_KEY = "172654867169949"
CLOUDINARY_API_SECRET = "Vre4sIxwv3my0QP3Knuq7QID55M"

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)
#token hugginface
user_input = input("Enter token: ")
login(token = user_input)
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'tommycik/controlFluxAlcolReduced'
#N4
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model,
    quantization_config=bnb_config,
    #device_map="auto" not supported
)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")

control_image = load_image("controlnet_dataset/imagesControlCanny/sample_0000_canny.jpg")
def main():
    user_input = input("Enter prompt: ")
    #prompt = "A tall glass with gemstones"
    prompt = user_input
    image = pipe(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.2,
        num_inference_steps=50,
        guidance_scale=6.0,
    ).images[0]
    image.save("image.jpg")
    #su a100 richiede 1 minuto inferenza
    try:
        # Save image to a BytesIO buffer
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG') # Or 'PNG' if you prefer
        img_byte_arr.seek(0) # Rewind to the beginning of the buffer

        # Generate a unique public ID for the image
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        public_id = f"generated_images/{timestamp}_{unique_id}" # Optional: organize in a folder

        # Upload to Cloudinary
        response = cloudinary.uploader.upload(
            img_byte_arr,
            public_id=public_id,
            folder="flux_controlnet_reduced_results", # Optional: specify a folder in Cloudinary
            resource_type="image"
        )

        print(f"Image uploaded to Cloudinary: {response['secure_url']}")

    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")

    print("Image pushed to Cloudinary successfully.")

if __name__ == '__main__':
    main()
