import torch
from huggingface_hub import login, hf_hub_download
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from peft import PeftModel, LoraConfig
from safetensors import safe_open
import os
import torchvision.transforms as T # Import torchvision transforms

# Login to Huggingface
user_input = input("Enter token: ")
login(token=user_input)

# Base and LoRA models on HF Hub
base_controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Canny"
lora_weights_repo = "tommycik/controlFluxAlcol-LoRAReduced"
base_flux_model = "black-forest-labs/FLUX.1-dev"

# Define the local path where you want to save the downloaded LoRA weights
local_lora_path = "./lora_weights"
os.makedirs(local_lora_path, exist_ok=True)

# Download the LoRA weights file
# Based on your provided file list, 'diffusion_pytorch_model.safetensors' is at the root.
lora_filename = hf_hub_download(repo_id=lora_weights_repo, filename="diffusion_pytorch_model.safetensors")

# Load base ControlNet model
controlnet = FluxControlNetModel.from_pretrained(base_controlnet_model, torch_dtype=torch.bfloat16)

# Define a dummy LoraConfig. These parameters (r, lora_alpha, target_modules)
# are critical and need to match how your LoRA was trained.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v",
            "add_q_proj", "add_k_proj", "add_v_proj",
            "to_out.0", "to_add_out"], # Common targets for attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
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
controlnet.load_state_dict(lora_state_dict, strict=False)

# Load the pipeline with the adapted ControlNet
pipe = FluxControlNetPipeline.from_pretrained(base_flux_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Load control image
control_image = load_image("controlnet_dataset/imagesControlCanny/sample_0000_canny.jpg")

# Preprocess the control image to a tensor
preprocess_transform = T.Compose([
    T.ToTensor(), # Converts PIL Image to Tensor (0-1 range)
    T.Normalize([0.5], [0.5]), # Normalizes to [-1, 1]
])
control_image_tensor = preprocess_transform(control_image).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

# Prompt input
user_prompt = input("Enter prompt: ")

# Generate image
image = pipe(
    user_prompt,
    control_image=control_image_tensor, # Pass the preprocessed tensor
    controlnet_conditioning_scale=[0.2], # Must be a list
    num_inference_steps=50,
    guidance_scale=6.0,
).images[0]

# Save and display
image.save("image.jpg")
image.show()