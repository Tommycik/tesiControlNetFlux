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


# Load base ControlNet model
# You might need to adjust these parameters based on how the LoRA was trained.
# These are common default values.
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices
    lora_alpha=32, # LoRA alpha parameter
    target_modules=["to_q", "to_k", "to_v",
            "add_q_proj", "add_k_proj", "add_v_proj",
            "to_out.0", "to_add_out"], # Adjust based on your LoRA's target modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM", # This might need to be adjusted based on the actual task, but often not critical for inference
)

# Initialize PeftModel with the base model and the dummy config
controlnet = PeftModel(controlnet, lora_config)

# Load LoRA weights from the repository
controlnet.load_adapter(lora_weights_repo + "/controlnet_lora", adapter_name="default")


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