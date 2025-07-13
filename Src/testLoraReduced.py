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

        # Load components for the base FLUX model individually.
        # This is necessary because the `from_pretrained` for the pipeline is failing.
        print("Loading tokenizer, text_encoder, vae, and scheduler from base FLUX model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = AutoModel.from_pretrained(base_model_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        # The UNet is the problematic component. Given previous errors,
        # it's likely not in a 'unet' subfolder or directly at the root with a config.json.
        # However, for a StableDiffusionControlNetPipeline, UNet is mandatory.
        # We will try to load it from the base_model_path without a subfolder,
        # as some models have the UNet config at the root level.
        # If this fails with config.json not found, it means black-forest-labs/FLUX.1-dev
        # truly does not expose its UNet in a way UNet2DConditionModel.from_pretrained expects.
        print("Attempting to load UNet from base FLUX model...")
        try:
            unet = UNet2DConditionModel.from_pretrained(base_model_path)
        except OSError as e:
            print(f"Failed to load UNet directly from {base_model_path}: {e}")
            print(
                "This suggests the FLUX model's UNet is not structured for direct loading by UNet2DConditionModel.from_pretrained.")
            print(
                "You might need to consult the 'black-forest-labs/FLUX.1-dev' documentation or source code for proper UNet loading,")
            print("or use a base model that is fully compatible with StableDiffusionControlNetPipeline.")
            return  # Exit if UNet cannot be loaded

        # Load the base ControlNet model first
        print("Loading base ControlNet model...")
        base_controlnet_pretrained_path = 'InstantX/FLUX.1-dev-Controlnet-Canny'
        controlnet = ControlNetModel.from_pretrained(base_controlnet_pretrained_path)

        # Apply the LoRA weights to the ControlNet model
        print(f"Loading Diffusers-style LoRA weights from {controlnet_lora_path} into ControlNet...")
        controlnet.load_lora_weights(controlnet_lora_path)

        # Create the StableDiffusionControlNetPipeline by explicitly passing all components
        print("Creating StableDiffusionControlNetPipeline with all components...")
        pipe = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,  # UNet is now explicitly passed after individual loading attempt
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

if __name__ == "__main__":
    main()