# MiniMind - Echo AI Friend

基于 Qwen2.5-3B-Instruct 微调的个性化AI朋友。

## 人设设定

- **名字**: Echo
- **用户名**: Never
- **关系**: 死党/损友，玩得很熟
- **性格**: 活泼幽默，爱开玩笑，嘴损心好
- **说话风格**: 口语化，像朋友聊天

## 用户画像

- **名字**: Never
- **学校**: 南航研究生
- **专业**: BME (生物医学工程)
- **年龄**: 22岁
- **爱好**: CSGO2、编程、游戏

## 工具能力

- 天气查询 (get_weather)
- 搜索信息 (search_web)
- 计算转换 (calculate)

## 训练数据统计

| 数据类型 | 数量 | 文件 |
|---------|------|------|
| 人设对话 | 118条 | persona.jsonl |
| 工具调用 | 100条 | tools.jsonl |
| 爱好相关 | 131条 | hobbies.jsonl |
| **总计** | **349条** | - |

## 项目结构

```
minimind/
├── data/                # 训练数据
│   ├── persona.jsonl    # 人设对话数据
│   ├── tools.jsonl      # 工具调用数据
│   └── hobbies.jsonl    # 爱好相关数据
├── scripts/             # 训练脚本
│   ├── train.py         # 训练入口
│   ├── merge_lora.py    # 合并权重
│   └── chat.py          # 聊天测试
├── configs/             # 配置文件
│   └── lora_config.yaml # LoRA参数
├── tools/               # 工具定义
│   └── tool_schema.json # 工具schema
├── models/              # 模型存放
└── requirements.txt     # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 开始训练

```bash
cd C:\Users\A\Desktop\minimind
python scripts/train.py
```

### 3. 合并权重

```bash
python scripts/merge_lora.py
```

### 4. 聊天测试

```bash
python scripts/chat.py
```

## 训练参数

- **基础模型**: Qwen/Qwen2.5-3B-Instruct
- **LoRA秩 (r)**: 16
- **学习率**: 2e-4
- **批大小**: 4 (gradient accumulation: 4)
- **训练轮数**: 3
- **显存需求**: ~6GB

## 注意事项

1. 确保GPU显存至少8GB
2. 首次运行会自动下载Qwen模型
3. 训练完成后权重保存在 `models/qwen-echo-friend-lora/`
4. 合并后的模型保存在 `models/qwen-echo-friend-merged/`
