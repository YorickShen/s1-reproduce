import sys
from pathlib import Path

# 确保在任意工作目录下运行都能正确定位项目根目录
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import S1TrainConfig
from src.dataset import load_s1_dataset


# 加载分词器
def get_tokenizer(config: S1TrainConfig):
    print(f"[*] 正在加载 Tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# QLoRA（4-bit 量化加载）模型初始化
def get_model(config: S1TrainConfig):
    # 声明模型如何被压缩与如何计算
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 模型实例化
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 冻结非量化参数并支持梯度检查点
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 配置 LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print("[*] LoRA 注入完毕，当前可训参数量:")
    model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    print("=== 正在运行模型与分词器加载自测 ===")
    config = S1TrainConfig()

    tokenizer = get_tokenizer(config)
    model = get_model(config)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[*] 显存占用 -> 已分配: {allocated:.2f} GB | 缓存保留: {reserved:.2f} GB")
        print("[✓] 积木 1 验证成功：模型与 LoRA 成功装配！")