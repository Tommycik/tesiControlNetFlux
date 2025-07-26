from huggingface_hub import login
import os
import subprocess
import shutil


def main():
    # token hugginface da tastiera
    user_input = input("Enter token: ")
    login(token=user_input)

    # Percorsi dataset e output
    output_dir = "model"

    # Nome base modello
    pretrained_model = "black-forest-labs/FLUX.1-dev"
    controlnet_pretrained = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    # Script ufficiale diffusers per il training
    training_script = "train_control_lora_official.py"  # Changed script name

    # Comando per chiamare lo script di training
    command = [
        "accelerate", "launch", training_script,
        "--pretrained_model_name_or_path", pretrained_model,
        #"--controlnet_model_name_or_path", controlnet_pretrained,
        "--output_dir", output_dir,
        "--conditioning_image_column", "condition_image",
        "--image_column", "image",
        "--caption_column", "prompt",
        "--jsonl_for_train", "./controlnet_dataset/dataset.jsonl",
        "--resolution", "256",
        "--learning_rate", "2e-6",
        "--max_train_steps", "2",
        "--checkpointing_steps", "250",
        "--validation_steps", "125",
        "--mixed_precision", "fp16",
        "--validation_image", "controlnet_dataset/images/sample_0000.jpg",
        "--validation_prompt",
        "transparent glass on white background, the bottom part of the glass presents light grooves ",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",
        # LoRA-specific
        "--use_lora",
        "--rank","2",# "16",
        #"--lora_alpha", "16", # "64",
        #"--lora_dropout", "0.1",
        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFluxAlcolLora",
        # Added LoRA specific arguments (optional, defaults in official script are often good)
        "--rank", "4",
        "--lora_layers", "all-linear"  # Example of explicit lora layers
    ]

    print("Esecuzione comando Accelerate:")
    print(" ".join(command))

    result = subprocess.run(command)


if __name__ == "__main__":
    main()
