from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import(
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedModel,
    PreTrainedTokenizer,
) 

# Budget Forcing 推理控制超参数
@dataclass
class BudgetForcingConfig:
    # 思考预算：期望模型在得到最终答案前至少产生思考的token数量
    thinking_budget: int = 512

    # 思考强行转折提示词
    turn_prompt: str = "\nWait, let me double check this ...\n"

    # 上限
    max_new_tokens: int = 1536

    # 标记定义
    think_start_token: str = "<|im_start|>think\n"
    answer_start_token: str = "<|im_start|>answer\n"
    eos_token: str = "<|im_end|>"

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
            # 预算已满如果还没有停止标记，直接手动停止思考
            if not torch.equal(current_ids[0, -len(answer_tokens):], stop_criteria.target_tensor):
                answer_tensor = torch.tensor([answer_tokens], dtype=torch.long, device=device)
                current_ids = torch.cat([current_ids, answer_tensor], dim=1)
            print(f"[*] 思考预算达成 （已思考 {thinking_tokens_count} tokens，拦截 {intercept_count}次），准备进入最终作答...")
            break

        # 还剩多少思考 token
        remaining_budget = config.thinking_budget - thinking_tokens_count

        ouputs = model.generate(
            input_ids=current_ids,
            max_new_tokens=remaining_budget,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
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
                break
        else:
            pass

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
    generated_tokens = final_outputs[0, prompt_len]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    return full_response

if __name__ == "__main__":
    from src.config import S1TrainConfig
    from src.train import get_tokenizer, get_model
    from src.dataset import format_s1_prompt_and_response

    print("=" * 60)
    print("[*] 正在载入 4-bit 模型与 Tokenizer 进行 Budget Forcing 推理冒烟实测...")
    print("=" * 60)

    cfg = S1TrainConfig()
    tokenizer = get_tokenizer(cfg)
    model = get_model(cfg)

    # 构造一道测试题
    test_question = "If 2x + 5 = 17, what is the value of x? Solve step by step."
    formatted = format_s1_prompt_and_response(
        question=test_question,
        thinking_trajectory="",
        attempt=""
    )
    # 我们只需要 Prompt 部分（以 <|im_start|>think\n 结尾）
    prompt = formatted["prompt"]

    print("\n[Input Prompt]:")
    print(prompt)

    # 实例化推理配置：要求模型至少深度思考 256 个 Token 才准交卷！
    forcing_config = BudgetForcingConfig(
        thinking_budget=256,
        turn_prompt="\nWait, let me rethink and double check my calculation step by step:\n"
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