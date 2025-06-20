import torch
import os
from huggingface_hub import login
os.environ["HF_TOKEN"] = "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"
os.environ['HF_HOME'] = 'modle'
token = os.environ["HF_TOKEN"]

import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel

login(token='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', add_to_git_credential=True)
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

control_image = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/canny.jpg")
prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=1,
    guidance_scale=3.5,
).images[0]
image.save("image.jpg")
image.show()
