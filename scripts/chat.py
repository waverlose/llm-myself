import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chat():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "qwen-echo-friend-merged"
    )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,  # 替换 torch_dtype
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print("模型加载失败，请检查模型文件是否完整。")
        print(f"错误信息: {e}")
        return
    model.eval()

    print("hi, I am Echo! (输入 'exit' 退出)")
    print("-" * 50)

    messages = [
        {"role": "system", "content": "你是Echo，Never的朋友，永远支持Never，永远陪伴Never。"}
    ]

    while True:
        user_input = input("Never: ")
        if user_input.lower() == "exit":
            print("Echo: 再见")
            break
        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"Echo: {response}")

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat()
