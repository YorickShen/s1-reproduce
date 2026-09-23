import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import(
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
) 
from peft import PeftModel

from src.config import S1TrainConfig
from src.dataset import format_s1_prompt_and_response

# Budget Forcing 推理控制超参数
@dataclass
class BudgetForcingConfig:
    # 思考预算：期望模型在得到最终答案前至少产生思考的token数量
    thinking_budget: int = 256

    # 思考强行转折提示词
    turn_prompt: str = "\nWait, let me double check this ...\n"

    # 上限
    max_new_tokens: int = 1536

    # 标记定义
    think_start_token: str = "<|im_start|>think\n"
    answer_start_token: str = "\n<|im_start|>answer\n"
    eos_token: str = "<|im_end|>"

    # 方案 2 核心：强引导交卷前缀（让模型在作答舱直奔标答）
    answer_lead_in: str = "Therefore, the final answer is \\boxed{"

    # 前向推理最大步长（默认与预算对齐，实现连贯推导，仅在模型主动交卷时抓包拦截）
    step_chunk_size: int = 1250

    # 最小反思保护窗口（低于此配额时不打断，自然收敛）
    min_rethink_window: int = 256

# 哨兵类，在目标 token 序列停止
class StopOnTokenSequenceCriteria(StoppingCriteria):
    def __init__(self, target_sequence: List[int], device: torch.device):
        super().__init__()
        self.seq_len = len(target_sequence)
        # 预先将目标token列表固化到GPU显存张量中
        self.target_tensor = torch.tensor(target_sequence, dtype=torch.long, device=device)

    # 回调函数
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 如果当前生成的总长度小于目标序列长度，直接放行
        if input_ids.shape[1] < self.seq_len:
            return False

        # 取序列最末端的 seq_len 个 token,在GPU内部做比对
        tail = input_ids[0,-self.seq_len:]
        return torch.equal(tail, self.target_tensor)

#  加载纯净量化基座模型，不提前包裹未经训练的 LoRA 壳
def load_clean_base_model(config: S1TrainConfig):
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        
    return model

# s1论文核心推理算法
def budget_forcing_generate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        config: Optional[BudgetForcingConfig] = None,
) -> str:
    if config is None:
        config = BudgetForcingConfig()

    device = model.device

    # 编码 prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = input_ids.shape[1]

    # 编码关键控制符号
    answer_tokens = tokenizer.encode(config.answer_start_token, add_special_tokens=False)
    turn_tokens = tokenizer.encode(config.turn_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    eos_token_id = tokenizer.encode(config.eos_token, add_special_tokens=False)[0]

    # 实例化抓包器
    stop_criteria = StopOnTokenSequenceCriteria(target_sequence=answer_tokens, device=device)
    stopping_criteria = StoppingCriteriaList([stop_criteria])

    current_ids = input_ids
    intercept_count = 0

    # 思考拦截循环
    while True:
        # 当前已经产生的思考 Token 数量
        thinking_tokens_count = current_ids.shape[1] - prompt_len

        # 是否达到思考预算
        if thinking_tokens_count >= config.thinking_budget:
            print(f"[*] 思考预算达成 （已思考 {thinking_tokens_count} tokens，拦截 {intercept_count}次），准备进入最终作答...")
            # 预算已满如果还没有停止标记，直接手动停止思考
            if not torch.equal(current_ids[0, -len(answer_tokens):], stop_criteria.target_tensor):
                answer_tensor = torch.tensor([answer_tokens], dtype=torch.long, device=device)
                current_ids = torch.cat([current_ids, answer_tensor], dim=1)
            
            # 方案 2 核心：注入强引导答题前缀（直奔标答格式）
            if config.answer_lead_in:
                lead_in_tokens = tokenizer.encode(config.answer_lead_in, add_special_tokens=False)
                lead_in_tensor = torch.tensor([lead_in_tokens], dtype=torch.long, device=device)
                current_ids = torch.cat([current_ids, lead_in_tensor], dim=1)
            break

        # 还剩多少思考 token
        remaining_budget = config.thinking_budget - thinking_tokens_count

        # 分段步长控制
        step_budget = min(remaining_budget, config.step_chunk_size)

        ouputs = model.generate(
            input_ids=current_ids,
            max_new_tokens=step_budget,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=False,
        )

        current_ids = ouputs

        # 检查末尾是否出现强制截停标记
        if torch.equal(current_ids[0, -len(answer_tokens):], stop_criteria.target_tensor):
            cur_thinking = current_ids.shape[1] - prompt_len - len(answer_tokens)
            if cur_thinking < config.thinking_budget:
                intercept_count += 1
                print(f"[*] [拦截抓包第 {intercept_count} 次] 模型在第 {cur_thinking} 个 token 交卷，当场截停...")

                # 剥离末尾的交卷标记
                current_ids = current_ids[:, :-len(answer_tokens)]

                # 强行注入转折词
                current_ids = torch.cat([current_ids, turn_tokens], dim=1)
                print(f"[*] 已强行注入思考转折词，继续思考推导")
            else:
                # 自然达到预算交卷，同样注入强引导
                if config.answer_lead_in:
                    lead_in_tokens = tokenizer.encode(config.answer_lead_in, add_special_tokens=False)
                    lead_in_tensor = torch.tensor([lead_in_tokens], dtype=torch.long, device=device)
                    current_ids = torch.cat([current_ids, lead_in_tensor], dim=1)
                break
        elif current_ids[0, -1].item() == eos_token_id:
            # 若模型输出了 <|im_end|>，剥除该标记并注入转折词
            intercept_count += 1
            print(f"[*] [拦截抓包第 {intercept_count} 次] 模型试图输出结束符退出，强制截断并注入反思词...")
            current_ids = current_ids[:,:-1]
            current_ids = torch.cat([current_ids, turn_tokens], dim=1)
        else:
            # 若本轮步长耗尽但未交卷，且总预算未满
            cur_thinking = current_ids.shape[1] - prompt_len
            if cur_thinking < config.thinking_budget:
                remaining = config.thinking_budget - cur_thinking
                # 末段保护窗口：若剩余预算不足以支撑一次完整的二次反思，不恶意打断
                if remaining >= config.min_rethink_window:
                    intercept_count += 1
                    print(f"[*] [主动启发 {intercept_count} 次] 模型已推导 {cur_thinking} tokens，主动注入转折词...")
                    current_ids = torch.cat([current_ids, turn_tokens], dim=1)

    # 计算模型还能使用的剩余最大token配额
    total_generated_so_far = current_ids.shape[1] - prompt_len
    remaining_tokens = max(128, config.max_new_tokens - total_generated_so_far)

    # 放行模型生成最终答案，此时只在遇到 <|im_end|> 时停止
    final_outputs = model.generate(
        input_ids=current_ids,
        max_new_tokens=remaining_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        do_sample=False,
    )

    # 截取 prompt 之后生成的全部 token （包含完整思考链与最终答案）
    generated_tokens = final_outputs[0, prompt_len:]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    return full_response

if __name__ == "__main__":
    from src.train import get_tokenizer

    print("=" * 60)
    print("[*] 正在载入纯净基座并单层挂载 LoRA 适配器权重...")
    print("=" * 60)

    cfg = S1TrainConfig()
    tokenizer = get_tokenizer(cfg)
    base_model = load_clean_base_model(cfg)

    checkpoint_dir = PROJECT_ROOT / "outputs" / "s1-7b-qlora" / "checkpoint-5"
    if checkpoint_dir.exists():
        print(f"[*] 注入已微调权重: {checkpoint_dir}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    else:
        print(f"[!] 警告: 未检测到 {checkpoint_dir}，使用原生基座")
        model = base_model

    # 切换至评估模式，冻结模型状态
    model.eval()

    # 构造一道测试题
    test_question = "Find the sum of all positive integers n such that n^2 + 19n + 48 is a perfect square. Show your detailed reasoning step by step."
    formatted = format_s1_prompt_and_response(
        question=test_question,
        thinking_trajectory="",
        attempt="",
    )
    prompt = formatted["prompt"]

    print("\n[Input Prompt]:")
    print(prompt)

    forcing_config = BudgetForcingConfig(
        thinking_budget=384,
        turn_prompt="\nWait, let me rethink this problem from another angle and verify my steps:"
    )

    print("\n[*] 正在启动 Budget Forcing 推理生成...")
    result = budget_forcing_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=forcing_config
    )

    print("\n" + "=" * 60)
    print("[*] 模型完整输出轨迹 (思考过程 + 注入转折 + 最终答案):")
    print("=" * 60)
    print(result)