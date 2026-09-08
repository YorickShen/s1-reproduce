from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

from typing import Dict, Optional, Any

'''
检视数据
raw_ds = load_dataset("simplescaling/s1K-1.1", split="train")
print(raw_ds[0].keys())
print(raw_ds[0]["question"])
'''

# 处理prompt和response
def format_s1_prompt_and_response(
        question: str,
        thinking_trajectory: str,
        attempt: str,
        system_prompt: str = "You are a helpful assistant."
) -> Dict[str, str]:

        # prompt文本
        prompt_txt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>think\n"  # 将开始思考划在prompt里，"自然而然地开始思考"
        )

        # Response文本
        response_txt = (
                f"{thinking_trajectory.strip()}\n"
                f"<|im_start|>answer\n"
                f"{attempt.strip()}<|im_end|>"
        )

        return {
                "prompt": prompt_txt,
                "response": response_txt,
                "full_text": prompt_txt + response_txt
        }

# 数据集清洗与批处理
def load_s1_dataset(
        dataset_name: str = "simplescaling/s1k-1.1",
        cot_source: str = "gemini",
        split: str = "train",
        max_samples: Optional[int] = None       # 前N条数据，默认是None
) -> Dataset:

        # 载入数据集
        print(f"[*] 正在载入数据集: {dataset_name} (split: {split}) ...")
        raw_ds = load_dataset(dataset_name, split=split)

        # 当输入max_samples进行快速调试时
        if max_samples is not None and max_samples > 0:
                raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))
                print(f"[*] 已截取前 {len(raw_ds)} 条样本进行快速测试")

        # 思考链和答案的来源
        think_key = f"{cot_source}_thinking_trajectory"
        attempt_key = f"{cot_source}_attempt"

        formmated_data = []
        for item in raw_ds:
                # 提取数据集中的问题、思考和答案
                question = item["question"]
                thinking = item.get(think_key, "")
                attempt = item.get(attempt_key, "")

                # 兜底：如果选定的cot字段为空，回退到solution
                if not thinking:
                        thinking = item.get("solution", "")
                if not attempt:
                        attempt = item.get("solution", "")

                formatted = format_s1_prompt_and_response(question, thinking, attempt)
                formmated_data.append({
                        "question": question,
                        "thinking": thinking,
                        "attempt": attempt,
                        "prompt": formatted["prompt"],
                        "response": formatted["response"],
                        "full_text": formatted["full_text"]
                })

        return Dataset.from_list(formmated_data)

# tokenize 与打掩码
def tokenize_s1_sample(
        example: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_lenth: int = 3072
) -> Dict[str, List[int]]:

        # 单独对 prompt 编码
        prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)

        # 单独对 response 编码
        response_ids = tokenizer.encode(example["response"], add_special_tokens=False)

        # 拼接
        input_ids = prompt_ids + response_ids

        # 给 prompt 打上掩码，-100
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) +list(response_ids)

        # 超过最大长度时截断
        if len(input_ids) > max_lenth:
                input_ids = input_ids[:max_lenth]
                attention_mask = attention_mask[:max_lenth]
                labels = labels[:max_lenth]

        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
        }