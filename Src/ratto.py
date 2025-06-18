from huggingface_hub import login
from datasets import load_dataset
import os
import subprocess
os.environ["HF_TOKEN"] = "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"
token = os.environ["HF_TOKEN"]

login(token)
def main():
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
        "--max_train_steps", "50",
        "--checkpointing_steps", "10",
        "--validation_steps", "10",
        "--mixed_precision", "fp16",
        "--validation_image", "controlnet_dataset/sample_0000.jpg",
        "--validation_prompt", "red circle with blue background cyan circle with brown floral background",
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
    
    if result.returncode == 0:
        original_model_path = os.path.join(output_dir, "pytorch_model.bin")
        target_model_path = os.path.join(output_dir, "diffusion_pytorch_model.fp32.bin")
        if os.path.exists(original_model_path):
            os.rename(original_model_path, target_model_path)
            print(f"Model file renamed to: {target_model_path}")
        else:
            print("Warning: Expected model file 'pytorch_model.bin' not found.")
    else:
        print("Training script failed. Skipping model rename.")

if __name__ == "__main__":
    main()
