from huggingface_hub import login
import subprocess

def main():
    # Ask for Hugging Face token interactively
    user_input = input("Enter token: ")
    login(token=user_input)

    # Paths and model identifiers
    output_dir = "controlnet_lora_model"
    pretrained_model = "black-forest-labs/FLUX.1-dev"
    controlnet_pretrained = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    training_script = "train_control_lora_flux.py"

    # Accelerate training command with LoRA-specific args
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
        "--learning_rate", "1e-4",  # LoRA can use a slightly higher LR
        "--max_train_steps", "1000",
        "--checkpointing_steps", "250",
        "--validation_steps", "125",
        "--mixed_precision", "bf16",
        "--validation_image", "controlnet_dataset/sample_0000.jpg",
        "--validation_prompt", "transparent glass on white background, the bottom part of the glass presents light grooves",
        "--train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",

        # 🆕 LoRA-specific flags
        "--use_lora",
        "--lora_rank", "4",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",

        # Push to Hugging Face Hub
        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFluxAlcol-LoRA"
    ]

    print("Running Accelerate command:")
    print(" ".join(command))

    subprocess.run(command)

if __name__ == "__main__":
    main()