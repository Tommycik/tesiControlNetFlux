from huggingface_hub import login
from datasets import load_dataset
import os
import subprocess
os.environ["HF_TOKEN"] = "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"
os.environ['HF_HOME'] = 'modle'
token = os.environ["HF_TOKEN"]

login(token="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")
def main():
    # Percorsi dataset e output
    output_dir = "modle"
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
        "--conditioning_image_column", "conditioning_image",
        "--image_column", "image",
        "--caption_column", "prompt",
        "--jsonl_for_train", "./controlnet_dataset/dataset.jsonl",
        "--resolution", "512",
        "--learning_rate", "1e-5",
        "--max_train_steps", "10000",
        "--checkpointing_steps", "200",
        "--validation_steps", "100",
        "--mixed_precision", "fp16",
        "--validation_image", "controlnet_dataset/sample_0000.jpg",
        "--validation_prompt", "red circle with blue background cyan circle with brown floral background",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--set_grads_to_none",
        "--report_to", "wandb",
        "--push_to_hub",
    ]

    print("Esecuzione comando Accelerate:")
    print(" ".join(command))

    subprocess.run(command)

if __name__ == "__main__":
    main()
