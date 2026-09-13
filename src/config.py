from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# 锚定根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class S1TrainConfig:
    # 模型基座
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 4bit 量化配置
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda:[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 训练超参数
    output_dir: str = str(PROJECT_ROOT/"outputs"/"s1-7b-qlora")
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 5
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 1
    save_strategy: str = "epoch"
    bf16: bool = True
    max_steps: int = -1

