from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class BudgetForcingConfig:
    '''
    Budget Forcing 推理控制超参数
    '''
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

