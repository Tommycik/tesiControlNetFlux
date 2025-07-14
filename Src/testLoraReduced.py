import torch
from huggingface_hub import login, hf_hub_download
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from peft import PeftModel, LoraConfig
import os # Import os for path manipulation

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
# Ensure 'diffusion_pytorch_model.safetensors' is the correct filename within your repo
# And 'controlnet_lora' is the correct subfolder if it exists.
# If 'diffusion_pytorch_model.safetensors' is at the root of 'tommycik/controlFluxAlcol-LoRAReduced', remove subfolder='controlnet_lora'.
lora_filename = hf_hub_download(repo_id=lora_weights_repo, filename="diffusion_pytorch_model.safetensors", subfolder="controlnet_lora")

# Load base ControlNet model
controlnet = FluxControlNetModel.from_pretrained(base_controlnet_model, torch_dtype=torch.bfloat16)

# Define a dummy LoraConfig. Adjust these parameters based on how the LoRA was trained.
lora_config = LoraConfig(
    r=16,  # Rank of the LoRA matrices
    lora_alpha=16, # LoRA alpha parameter
    target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Adjust based on your LoRA's target modules
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Initialize PeftModel with the base model and the dummy config
controlnet = PeftModel(controlnet, lora_config)

# Load the state dictionary directly from the downloaded file
# Added weights_only=False to resolve the UnpicklingError
lora_state_dict = torch.load(lora_filename, weights_only=False) # Changed line
controlnet.load_state_dict(lora_state_dict, strict=False) # strict=False might be needed if there are minor mismatches

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