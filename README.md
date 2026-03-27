# 🤖 Create Your Own AI Friend

A simple project to fine-tune Qwen2.5-3B-Instruct with LoRA(every model), creating a personalized AI companion that can run locally on your computer.

## ✨ Features

- 🎯 Fine-tune with custom personality and conversation style
- 💬 Natural, casual conversation like talking to a friend
- 🔒 Run completely locally - no data leaves your machine
- ⚡ Optimized with LoRA for efficient training
- 💻 Works with LM Studio for easy local deployment

## 📋 Requirements

- Python 3.10+
- 16GB+ RAM (for CPU inference)
- OR NVIDIA GPU with 6GB+ VRAM (for faster training)
- OR AMD GPU (CPU inference supported)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/waverlose/llm-myself.git
cd llm-myself
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Training Data

Create your training data in `data/your_training_data.jsonl`:

```jsonl
{"messages": [{"role": "system", "content": "You are Echo, a friendly AI companion."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hey there! I'm Echo, nice to meet you!"}]}
{"messages": [{"role": "system", "content": "You are Echo, a friendly AI companion."}, {"role": "user", "content": "What do you like?"}, {"role": "assistant", "content": "I love chatting with friends and learning new things!"}]}
```

See `data/example_data.jsonl` for the format.

**Tips for good training data:**
- Include 200+ conversation examples
- Cover different topics and scenarios
- Use consistent personality/language style
- Include both questions and answers

### 4. Start Training

```bash
python scripts/train_fast.py
```

Training takes about 10-15 minutes on RTX 4090, or 1-2 hours on slower GPUs.

### 5. Merge LoRA Weights

```bash
python scripts/merge_lora.py
```

### 6. Convert to GGUF Format

```bash
pip install sentencepiece
python convert_hf_to_gguf.py models/qwen-echo-friend-merged --outtype f16 --outfile models/your-ai-friend.gguf
```

### 7. Run with LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Copy `models/your-ai-friend.gguf` to LM Studio's models folder
3. Load the model and start chatting!

## 📁 Project Structure

```
llm-myself/
├── scripts/
│   ├── train_fast.py      # Training script with LoRA
│   ├── merge_lora.py      # Merge LoRA weights
│   ├── chat.py            # Local chat testing
│   └── check_gpu.py       # Check GPU status
├── data/
│   └── example_data.jsonl # Example training format
├── convert_hf_to_gguf.py  # Convert to GGUF format
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## ⚙️ Training Parameters

Edit `scripts/train_fast.py` to customize:

```python
# Model to fine-tune (change to your preferred model)
model_name = "Qwen/Qwen2.5-3B-Instruct"

# LoRA configuration
lora_config = LoraConfig(
    r=16,              # LoRA rank (8-32)
    lora_alpha=32,     # LoRA scaling
    lora_dropout=0.05, # Dropout rate
)

# Training configuration
training_args = TrainingArguments(
    num_train_epochs=2,           # Training epochs
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    # ... more options
)
```

## 🎨 Customization Ideas

### Personality Examples

```jsonl
// Casual friend
{"messages": [{"role": "system", "content": "You are a chill friend who speaks casually, uses slang, and loves to joke around."}, ...]}

// Professional assistant
{"messages": [{"role": "system", "content": "You are a professional assistant who speaks formally and precisely."}, ...]}

// Study buddy
{"messages": [{"role": "system", "content": "You are an encouraging study buddy who helps with learning and stays positive."}, ...]}
```

### Topics to Include

- Daily greetings and small talk
- Hobbies and interests
- Subject-specific knowledge
- Emotional support conversations
- Problem-solving discussions

## 💡 Tips

1. **Data Quality > Quantity**: 500 high-quality examples beat 2000 low-quality ones
2. **Consistency**: Keep the personality consistent across all training examples
3. **Test Iteratively**: Train small, test, adjust, repeat
4. **Backup**: Save your training data and model checkpoints

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# In train_fast.py, reduce batch size or sequence length
per_device_train_batch_size=1
max_seq_length=512  # Add this if not present
```

### Model Not Loading
- Ensure GGUF file is complete and not corrupted
- Check LM Studio model folder path
- Try re-converting with `--outtype q4_k_m` for smaller size

### AMD GPU (A卡) Support
The project supports AMD GPUs through CPU inference. No special setup needed.

## 📜 License

MIT License

## 🙏 Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen2.5) for the base model
- [Hugging Face](https://huggingface.co/) for transformers library
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion
- [LM Studio](https://lmstudio.ai/) for local inference

## 📬 Contact

- GitHub: [@waverlose](https://github.com/waverlose)

---

**Made with ❤️ for everyone who wants their own AI friend**
