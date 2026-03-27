import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save():
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    lora_path = "./models/qwen-echo-friend-lora"
    output_path = "./models/qwen-echo-friend-merged"

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


if __name__ == "__main__":
    merge_and_save()
