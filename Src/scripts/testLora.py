import torch
from huggingface_hub import login
from controlnet_aux import OpenposeDetector
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from peft import PeftModel  # You need to have peft installed: pip install peft
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
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


# Login to Huggingface
user_input = input("Enter token: ")
login(token=user_input)

# Base and LoRA models on HF Hub
base_controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
lora_weights_repo = "tommycik/controlFluxAlcolLoRA"
base_flux_model = "black-forest-labs/FLUX.1-dev"

# Load base ControlNet model
controlnet = FluxControlNetModel.from_pretrained(base_controlnet_model, torch_dtype=torch.bfloat16)

# Load LoRA weights and merge into base controlnet
controlnet = PeftModel.from_pretrained(controlnet, lora_weights_repo)

# Load the pipeline with the adapted ControlNet
pipe = FluxControlNetPipeline.from_pretrained(base_flux_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

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