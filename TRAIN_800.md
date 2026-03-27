# 训练指南 - 800条数据版本

## 数据概况

已生成 **800条** 高质量训练数据，包含：

| 类别 | 内容 | 特点 |
|------|------|------|
| 身份认知 | Echo/Never名字、关系 | **确保记住名字** |
| 日常闲聊 | 生活、心情、日常 | 自然对话风格 |
| CSGO2 | 游戏技巧、策略 | 你爱玩的内容 |
| 编程 | Python、算法、效率 | 技术话题 |
| 研究生生活 | 实验压力、论文、就业 | BME相关 |
| 人生哲学 | 意义、成功、压力 | 深度话题 |
| 情感支持 | 难过、孤独、鼓励 | 朋友般的陪伴 |

## 训练步骤

### Step 1: 打开 Anaconda Prompt

### Step 2: 激活环境
```bash
conda activate minimind
cd C:\Users\A\Desktop\minimind
```

### Step 3: 安装CUDA版PyTorch（重要！）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: 验证CUDA
```bash
python scripts/check_gpu.py
```
应该显示：`CUDA available: True`

### Step 5: 开始训练
```bash
python scripts/train_fast.py
```

## 预期输出

```
[1/6] Loading tokenizer...
[2/6] Loading model...
[3/6] Configuring LoRA...
[4/6] Loading training data...
Loaded 800 examples
Data includes: identity, daily, CSGO2, coding, grad-life, philosophy, emotional
[5/6] Configuring training...
[6/6] Starting training...
```

## 训练时间

- 模型下载：约10分钟
- 训练：约1-3小时（取决于GPU）

## 训练完成后

### 合并权重
```bash
python scripts/merge_lora.py
```

### 测试聊天
```bash
python scripts/chat.py
```

## 数据文件位置

- `data/merged_training.jsonl` - 800条训练数据

## 如果遇到问题

### CUDA不可用
检查NVIDIA驱动是否最新，确保支持CUDA 12.1

### 显存不足
修改 `scripts/train_fast.py`:
- `max_seq_length=512`（从1024减少）
- `per_device_train_batch_size=1`（已经是1）
