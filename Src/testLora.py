import torch
from huggingface_hub import login
from controlnet_aux import OpenposeDetector
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import cloudinary
import cloudinary.uploader
import uuid
import subprocess
from datetime import datetime
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
# Login to Huggingface
user_input = input("Enter token: ")
login(token=user_input)

# Base and LoRA models on HF Hub
lora_weights_repo = "tommycik/controlFluxAlcolLoRA"
base_flux_model = "black-forest-labs/FLUX.1-dev"

transformer = FluxTransformer2DModel.from_pretrained(lora_weights_repo)
pipe = FluxControlPipeline.from_pretrained(
  base_flux_model,  transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Load control image
control_image = load_image("controlnet_dataset/imagesControlCanny/sample_0000_canny.jpg")

def main():
    # Prompt input
    user_prompt = input("Enter prompt: ")

    # Generate image
    image = pipe(
        user_prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.2,
        num_inference_steps=50,
        guidance_scale=6.0,
    ).images[0]

    # Save and display
    image.save("image.jpg")
    image.show()
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
            folder="flux_control_lora_results", # Optional: specify a folder in Cloudinary
            resource_type="image"
        )

        print(f"Image uploaded to Cloudinary: {response['secure_url']}")

    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")

    print("Image pushed to Cloudinary successfully.")
if __name__ == '__main__':
    main()