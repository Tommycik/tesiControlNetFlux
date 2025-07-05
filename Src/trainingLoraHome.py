from huggingface_hub import login
import subprocess

def main():
    # Ask for Hugging Face token interactively
    user_input = input("Enter token: ")
    login(token=user_input)

    # Paths and model identifiers
    output_dir = "controlnet_lora_model_8gb"
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

        # 🧠 Lower memory resolution
        "--resolution", "256",  # ⬅️ Down from 512 or 384

        # ⏱ Conservative runtime
        "--learning_rate", "5e-5",
        "--max_train_steps", "800",  # Reduce training time too
        "--checkpointing_steps", "200",
        "--validation_steps", "400",  # Less frequent validation

        # 🔧 Lower precision
        "--mixed_precision", "fp16",  # Smaller than bf16

        # ✅ Only one validation image/prompt to reduce inference VRAM
        "--validation_image", "controlnet_dataset/images/sample_0000.jpg",
        "--validation_prompt",
        "transparent glass on white background, the bottom part of the glass presents light grooves",

        # 🧪 Micro-batch + accumulation = small memory footprint
        "--train_batch_size", "1",  # Small batch
        "--gradient_accumulation_steps", "16",  # Simulate larger batch

        # 🔁 Memory optimizations
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",

        # 🪶 LoRA config
        "--use_lora",
        "--lora_rank", "4",
        "--lora_alpha", "16",  # Slightly smaller for memory
        "--lora_dropout", "0.05",  # Lower dropout for better learning in smaller runs

        # 🌐 Optional push to hub
        "--push_to_hub",
        "--hub_model_id", "tommycik/controlFluxAlcol-LoRA-8gb"
    ]

    print("Running Accelerate command:")
    print(" ".join(command))

    subprocess.run(command)

if __name__ == "__main__":
    main()