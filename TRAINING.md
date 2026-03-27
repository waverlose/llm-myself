# MiniMind 训练完整指南

## 当前状态

项目已准备就绪，包括：
- ✅ 349条训练数据（persona、tools、hobbies）
- ✅ Anaconda环境已创建（minimind）
- ✅ 所有依赖已安装
- ⚠️ PyTorch CUDA版本需要确认

## 训练步骤

### 方式1: 使用批处理脚本（推荐）

打开 **Anaconda Prompt**，执行：

```bash
cd C:\Users\A\Desktop\minimind
scripts\train_windows.bat
```

### 方式2: 手动执行

```bash
# 1. 打开Anaconda Prompt
# 2. 激活环境
conda activate minimind

# 3. 检查CUDA
python scripts/check_gpu.py

# 4. 如果CUDA不可用，安装CUDA版本PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 再次检查CUDA
python scripts/check_gpu.py

# 6. 开始训练
python scripts/train_fast.py
```

## 预期输出

训练过程中会显示：
- 加载模型进度
- LoRA参数信息
- 训练损失变化
- 模型保存位置

训练完成后，模型保存在：
- `models/qwen-echo-friend-lora/` - LoRA权重
- `models/qwen-echo-friend/` - 检查点

## 训练时间估算

- 首次下载模型: 5-10分钟（约6GB）
- 实际训练: 1-3小时（取决于GPU）

## 合并权重

训练完成后：

```bash
conda activate minimind
python scripts/merge_lora.py
```

## 测试聊天

```bash
conda activate minimind
python scripts/chat.py
```

## 问题排查

### Q: CUDA不可用？
A: 安装CUDA版本PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q: 显存不足？
A: 修改 `scripts/train_fast.py`:
- 减少 `per_device_train_batch_size` 到 1
- 减少 `max_seq_length` 到 512

### Q: 训练很慢？
A: 确保使用GPU:
```bash
python scripts/check_gpu.py
```
应该显示 `CUDA available: True`
