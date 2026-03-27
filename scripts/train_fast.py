import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def load_data(file_paths):
    all_data = []
    for path in file_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
    return all_data


def format_chat(example, tokenizer):
    messages = example.get("messages", [])
    formatted = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            if content is None:
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_call_str = json.dumps(tool_calls, ensure_ascii=False)
                    formatted += f"<|im_start|>assistant\n{tool_call_str}<|im_end|>\n"
            else:
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            tool_content = msg.get("content", "")
            formatted += f"<|im_start|>tool\n{tool_content}<|im_end|>\n"
    return formatted


def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print("=" * 50)
    print("MiniMind Training Script")
    print("=" * 50)

    print(f"\n[1/6] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded!")

    print("\n[2/6] Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded!")

    print("\n[3/6] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/6] Loading training data...")
    data_files = [
        "./data/merged_training.jsonl",
    ]
    all_data = load_data(data_files)
    print(f"Loaded {len(all_data)} examples")
    print(
        f"Data includes: identity, daily, CSGO2, coding, grad-life, philosophy, emotional"
    )

    def preprocess_function(example):
        return {"text": format_chat(example, tokenizer)}

    dataset = Dataset.from_list(all_data)
    dataset = dataset.map(preprocess_function)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print("\n[5/6] Configuring training...")
    training_args = TrainingArguments(
        output_dir="./models/qwen-echo-friend",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        warmup_steps=20,
        save_total_limit=2,
        report_to="none",
        max_grad_norm=1.0,
    )

    print("\n[6/6] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print("\n" + "=" * 50)
    print("Training complete! Saving model...")
    print("=" * 50)
    model.save_pretrained("./models/qwen-echo-friend-lora")
    tokenizer.save_pretrained("./models/qwen-echo-friend-lora")
    print("\nModel saved to ./models/qwen-echo-friend-lora")
    print("Done!")


if __name__ == "__main__":
    main()
