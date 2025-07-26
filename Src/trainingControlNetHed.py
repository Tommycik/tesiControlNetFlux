from huggingface_hub import login
from datasets import load_dataset
import os
import subprocess
from safetensors.torch import load_file
import torch
import shutil

#login(token = "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")
def main():
    #token hugginface da tastiera
    user_input = input("Enter token: ")
    login(token = user_input)
    # Percorsi dataset e output
    output_dir = "model"
    #dataset = load_dataset("./controlnet_dataset/dataset.py", data_dir="./controlnet_dataset")
    #print(dataset["train"].column_names)
    # Nome base modello
    pretrained_model = "black-forest-labs/FLUX.1-dev"
    controlnet_pretrained = "Xlabs-AI/flux-controlnet-hed-diffusers"
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
        "--jsonl_for_train", "./controlnet_dataset/dataset_hed.jsonl",
        "--resolution", "512",
        "--learning_rate", "2e-6",
        "--max_train_steps", "2",
        "--checkpointing_steps", "250",
        "--validation_steps", "125",
        "--mixed_precision", "bf16",
        "--validation_image", "controlnet_dataset/images/sample_0000.jpg",
        "--validation_prompt", "transparent glass on white background, the bottom part of the glass presents light grooves ",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",
        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFluxAlcolHed"
    ]

    print("Esecuzione comando Accelerate:")
    print(" ".join(command))

    result = subprocess.run(command)
    

if __name__ == "__main__":
    main()
