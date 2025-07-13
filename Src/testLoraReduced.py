import torch
from huggingface_hub import login
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from peft import PeftModel  # You need to have peft installed: pip install peft

# Login to Huggingface
user_input = input("Enter token: ")
login(token=user_input)

# Base and LoRA models on HF Hub
base_controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
lora_weights_repo = "tommycik/controlFluxAlcol-LoRAReduced"
base_flux_model = "black-forest-labs/FLUX.1-dev"

# Load base ControlNet model
controlnet = FluxControlNetModel.from_pretrained(base_controlnet_model, torch_dtype=torch.bfloat16)

# Load LoRA weights and merge into base controlnet
controlnet = PeftModel.from_pretrained(PeftModel.from_pretrained(controlnet, lora_weights_repo/"controlnet_lora"))

# Load the pipeline with the adapted ControlNet
pipe = FluxControlNetPipeline.from_pretrained(base_flux_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load control image
control_image = load_image("controlnet_dataset/imagesControlCanny/sample_0000_canny.jpg")

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