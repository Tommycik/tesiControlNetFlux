from huggingface_hub import login
from datasets import load_dataset
import os
import subprocess
from safetensors.torch import load_file
import torch
import shutil

#login(token = "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")
def main():
    #token hugginface
    user_input = input("Enter token: ")
    login(token = user_input)
    # Percorsi dataset e output
    output_dir = "model"
    #dataset = load_dataset("./controlnet_dataset/dataset.py", data_dir="./controlnet_dataset")
    #print(dataset["train"].column_names)
    # Nome base modello
    pretrained_model = "black-forest-labs/FLUX.1-dev"
    controlnet_pretrained = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    # Script ufficiale diffusers per il training
    training_script = "train_controlnet_flux.py"
        # Comando per chiamare lo script di training
    command = [
        "accelerate", "launch", training_script,
        "--pretrained_model_name_or_path", pretrained_model,
        "--controlnet_model_name_or_path", controlnet_pretrained,
        "--output_dir", output_dir,
        "--conditioning_image_column", "condition_image",
        "--image_column", "image",
        "--caption_column", "prompt",
        "--jsonl_for_train", "./controlnet_dataset/dataset.jsonl",
        "--resolution", "512",
        "--learning_rate", "1e-5",
        "--max_train_steps", "2",
        "--checkpointing_steps", "1",
        "--validation_steps", "1",
        "--mixed_precision", "bf16",
        "--validation_image", "controlnet_dataset/sample_0000.jpg",
        "--validation_prompt", "transparent glass with stripes on the bottom on white background",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",
        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFlux"
    ]

    print("Esecuzione comando Accelerate:")
    print(" ".join(command))

    result = subprocess.run(command)
    
    # Load safetensors file
    safetensor_path = "model/diffusion_pytorch_model.safetensors"
    state_dict = load_file(safetensor_path)
    
    # Save as PyTorch .bin format
    torch.save(state_dict, "model/diffusion_pytorch_model.fp32.bin")
    
    # Also rename the safetensor file with the expected name
    shutil.copy(safetensor_path, "model/diffusion_pytorch_model.fp32.safetensors")
    
    print("✅ Conversion complete: both .fp32.safetensors and .fp32.bin generated.")

if __name__ == "__main__":
    main()
