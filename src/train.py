import sys
from pathlib import Path

# 确保在任意工作目录下运行都能正确定位项目根目录
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import S1TrainConfig
from src.dataset import load_s1_dataset, tokenize_s1_sample


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

# 构建动态批次整理器
def get_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

# 进度汇报
class PrettyProgressCallback(TrainerCallback):
     def on_log(self, args, state, control, logs=None, **kwargs):
          if logs and "loss" in logs:
               cur_step = state.global_step
               max_step = state.max_steps
               cur_epoch = logs.get("epoch", state.epoch or 0.0)
               loss = logs.get("loss", 0.0)
               lr = logs.get("learning_rate", 0.0)

               # 步数进度百分比
               percent = (cur_step / max_step * 100) if max_step > 0 else 0.0

               # 总 epoch 显示容错处理
               total_epochs = (
                    f"{args.num_train_epochs:.1f}"
                    if args.num_train_epochs is not None
                    else "?"
               )

               print(
                    f"[*] [Epoch {cur_epoch:.2f}/{total_epochs}] "
                    f"Step: {cur_step}/{max_step} ({percent:.1f}) |"
                    f"Loss: {loss:.4f} | LR: {lr:.2e}"
               )

# 训练总装函数
def train(config:S1TrainConfig, max_samples: int = None):
    # 准备核心组件
    tokenizer = get_tokenizer(config)
    model = get_model(config)
    data_collator = get_data_collator(tokenizer)

    # 加载数据并映射分词
    raw_ds = load_s1_dataset(max_samples=max_samples)
    print(f"[*] 正在执行分词与 Loss 掩码 (max_seq_length={config.max_seq_length})...")
    train_ds = raw_ds.map(
        lambda example: tokenize_s1_sample(
            example,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
        ),
        remove_columns=raw_ds.column_names,
        desc="[*] 正在分词与打掩码"
    )

    # 装配训练超参
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        optim=config.optim,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        max_steps=config.max_steps,
        save_strategy=config.save_strategy,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        report_to="None",
    )

    # 组装Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        callbacks=[PrettyProgressCallback()]
    )

    print("[*] 开始训练...")
    trainer.train()

    #训练完毕后保存 LoRA 适配器权重与配置
    print(f"[*] 训练完毕，正在保存模型权重至：{config.output_dir}...")
    trainer.save_model()
    print("[*] 模型保存成功!")

