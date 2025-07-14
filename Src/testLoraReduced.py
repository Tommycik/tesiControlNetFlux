import torch
from huggingface_hub import login, hf_hub_download
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from peft import PeftModel, LoraConfig
from safetensors import safe_open # Import safe_open
import os

# Login to Huggingface
user_input = input("Enter token: ")
login(token=user_input)

# Base and LoRA models on HF Hub
base_controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
lora_weights_repo = "tommycik/controlFluxAlcol-LoRAReduced"
base_flux_model = "black-forest-labs/FLUX.1-dev"

# Define the local path where you want to save the downloaded LoRA weights
local_lora_path = "./lora_weights"
os.makedirs(local_lora_path, exist_ok=True) # Create the directory if it doesn't exist

# Download the LoRA weights file
# We are downloading 'diffusion_pytorch_model.safetensors' directly.
# Based on the file list, it seems to be at the root of the repo, not in 'controlnet_lora'.
# If it was in a subfolder, you would use subfolder='your_subfolder_name'.
lora_filename = hf_hub_download(repo_id=lora_weights_repo, filename="diffusion_pytorch_model.safetensors") # Removed subfolder="controlnet_lora"

# Load base ControlNet model
controlnet = FluxControlNetModel.from_pretrained(base_controlnet_model, torch_dtype=torch.bfloat16)

# Define a dummy LoraConfig. Adjust these parameters based on how the LoRA was trained.
# These values (r, lora_alpha, target_modules) are crucial and need to match the LoRA training config.
# If you don't know them, you might need to infer them or check the original training script.
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Common targets for attention layers
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM", # This often doesn't strictly matter for inference but should ideally match
)

# Initialize PeftModel with the base model and the dummy config
controlnet = PeftModel(controlnet, lora_config)

# Load the state dictionary using safetensors library
lora_state_dict = {}
try:
    with safe_open(lora_filename, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    print("LoRA weights loaded successfully with safetensors.")
except Exception as e:
    print(f"Error loading LoRA weights with safetensors: {e}")
    print("Please ensure 'safetensors' library is installed (pip install safetensors) and the file is not corrupted.")
    raise # Re-raise the exception if safetensors fails

# Apply the loaded LoRA weights to the controlnet model
# PEFT expects certain key names for LoRA weights (e.g., 'base_model.model.to_q.lora_A.weight').
# Your 'diffusion_pytorch_model.safetensors' likely contains these correctly formatted keys.
controlnet.load_state_dict(lora_state_dict, strict=False) # strict=False can help if there are minor key mismatches

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
    controlnet_conditioning_scale=[0.2],
    num_inference_steps=50,
    guidance_scale=6.0,
).images[0]

# Save and display
image.save("image.jpg")
image.show()