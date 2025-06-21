import torch
import os
from huggingface_hub import login
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

#token hugginface
user_input = input("Enter token: ")
login(token = user_input)
base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'tommycik/controlFlux'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

control_image = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/canny.jpg")
user_input = input("Enter prompt: ")
#prompt = "A tall glass with gemstones"
prompt = user_input
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=15,
    guidance_scale=4.0,
).images[0]
image.save("image.jpg")
image.show()
