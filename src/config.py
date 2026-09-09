from dataclasses import dataclass, field
from typing import List

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