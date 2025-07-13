import os
from PIL import Image
import torch
from huggingface_hub import login
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, AutoencoderKL, DDIMScheduler, \
    UNet2DConditionModel, AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModel

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

    # This is the path to trained Diffusers-style ControlNet LoRA.
    controlnet_lora_path = "tommycik/controlFluxAlcol-LoRAReduced/controlnet_lora"

    # Load the base ControlNet model first
    print("Loading base ControlNet model...")
    base_controlnet_pretrained_path = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    controlnet = ControlNetModel.from_pretrained(base_controlnet_pretrained_path)

    # Attempt to load the StableDiffusionControlNetPipeline directly
    # by providing both the base FLUX model and the ControlNet model.
    # This is the most common way to initialize a ControlNet pipeline.
    print(
        f"Loading StableDiffusionControlNetPipeline from base model {base_model_path} and ControlNet {base_controlnet_pretrained_path}...")

    #The ControlNet model is passed directly as an argument to from_pretrained,
    # not extracted from a separate pipeline.
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16  # Often beneficial for performance, adjust if needed
    )

    # Apply the LoRA weights to the ControlNet model that's now part of the pipe
    print(f"Loading Diffusers-style LoRA weights from {controlnet_lora_path} into ControlNet...")
    # Access the controlnet from the loaded pipe instance
    pipe.controlnet.load_lora_weights(controlnet_lora_path)

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
        image=control_image,
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