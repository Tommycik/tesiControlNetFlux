from huggingface_hub import login
import subprocess
import os
import torch
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def main():
    # Ask for Hugging Face token interactively
    user_input = input("Enter token: ")
    login(token=user_input)

    # Paths and model identifiers
    output_dir = "controlnet_lora_model_reduced"
    pretrained_model = "black-forest-labs/FLUX.1-dev"
    controlnet_pretrained = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    training_script = "train_control_lora_flux_reduced.py"

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
        "--resolution", "128",  # ⬅️ reduced from 512
        "--learning_rate", "5e-5",
        "--max_train_steps", "2",
        "--checkpointing_steps", "250",
        "--validation_steps", "500",  # ⬅️ less frequent validation
        "--mixed_precision", "fp16",
        "--validation_image", "controlnet_dataset/images/sample_0000.jpg",
        "--validation_prompt", "transparent glass on white background, the bottom part of the glass presents light grooves",
        "--train_batch_size", "1",  # ⬅️ lower batch.
        "--gradient_accumulation_steps", "1",  # ⬅️ to maintain effective batch
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--enable_model_cpu_offload",
        "--set_grads_to_none",

        # LoRA-specific
        "--use_lora",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",

        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFluxAlcol-LoRAReduced"
    ]

    print("Running Accelerate command:")
    print(" ".join(command))

    subprocess.run(command)

if __name__ == "__main__":
    main()