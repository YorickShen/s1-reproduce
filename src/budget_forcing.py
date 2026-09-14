from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import StoppingCriteria

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
    answer_start_token: str = "<|im_start|answer\n>"
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