
# Base and LoRA models on HF Hub
from huggingface_hub import login
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, AutoencoderKL, DDIMScheduler
from diffusers import UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel
import torch
import os
from PIL import Image

base_controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
lora_weights_repo = "tommycik/controlFluxAlcol-LoRAReduced"
base_flux_model = "black-forest-labs/FLUX.1-dev"

def main():
    # Ask for Hugging Face token interactively if needed, or remove if not pushing to hub
    # Login to Huggingface
    user_input = input("Enter token: ")
    login(token=user_input)

    # Paths and model identifiers
    base_model_path = "black-forest-labs/FLUX.1-dev"

    # This is the path to your trained Diffusers-style ControlNet LoRA.
    controlnet_lora_path = "tommycik/controlFluxAlcol-LoRAReduced/controlnet_lora"

    # Load components
    print("Loading tokenizer, text_encoder, unet, vae from base model...")
    from diffusers import StableDiffusionPipeline
    flux_pipeline = StableDiffusionPipeline.from_pretrained(base_model_path)

    tokenizer = flux_pipeline.tokenizer
    text_encoder = flux_pipeline.text_encoder
    unet = flux_pipeline.unet
    vae = flux_pipeline.vae
    scheduler = flux_pipeline.scheduler

    # Apply the LoRA weights to the ControlNet model
    print(f"Loading Diffusers-style LoRA weights from {controlnet_lora_path} into ControlNet...")
    controlnet.load_lora_weights(controlnet_lora_path)

    # Create the StableDiffusionControlNetPipeline with the LoRA-infused ControlNet
    print("Creating StableDiffusionControlNetPipeline...")
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipe.to("cuda")

    # Example: Inference
    print("Performing inference...")
    prompt = "transparent glass on white background, the bottom part of the glass presents light grooves"

    control_image_path = "./controlnet_dataset/images/sample_0000.jpg"
    if not os.path.exists(control_image_path):
        print(f"Error: Control image not found at {control_image_path}")
        return

    control_image = Image.open(control_image_path).convert("RGB")
    control_image = control_image.resize((256, 256))

    generator = torch.Generator(device="cuda").manual_seed(0)

    controlnet_conditioning_scale = 0.7
    guidance_scale = 7.5

    output_image = pipe(
        prompt=prompt,
        image=control_image,  # StableDiffusionControlNetPipeline often uses 'image' for control image
        num_inference_steps=20,
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
    ).images[0]

    output_filename = "output_image_with_controlnet_lora_weighted.png"
    output_image.save(output_filename)
    print(f"Inference complete. Output saved to {output_filename}")


if __name__ == "__main__":
    main()