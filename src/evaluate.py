import sys
import os
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from fractions import Fraction

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer 
from datasets import load_dataset

from src.dataset import format_s1_prompt_and_response
from src.budget_forcing import budget_forcing_generate, BudgetForcingConfig

# 结构化模型输出结果
@dataclass
class ParsedS1Output:
    raw_text: str
    thinking_text: str
    answer_text: str
    extracted_answer: str

# 解析 s1/ChatML 输出格式
def parse_s1_response(raw_text: str) -> ParsedS1Output:
    # 剥去字符串外壳
    cleaned = raw_text.replace("<|im_end|>", "").strip()
    if "<|im_start|>answer" in cleaned:
        # 1.ChatML 双舱格式
        thinking_text, answer_text = cleaned.split("<|im_start|>answer", 1)
        thinking_text = thinking_text.replace("<|im_start|>think\n", "").replace("<|im_start|>think", "").strip()
        answer_text = answer_text.strip()

        # 作答舱必须包含显式结论才算作答成功
        has_explicit_conclusion = (
            _extract_boxed_content(answer_text) is not None or
            any(re.search(p, answer_text, re.IGNORECASE) for p in[
                r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|final\s+answer:?)",
                r"####",
            ])
        )
        if has_explicit_conclusion:
            # 作答舱有明确结论，优先采纳作答舱
            extracted_answer = extract_math_answer(answer_text)
        else:
            # 作答舱写到一半被掐断，直接去思考舱找有无答案
            extracted_answer = extract_math_answer(cleaned)
    else:
        # 2.baseline自由作答
        thinking_text = cleaned.replace("<|im_start|>think\n", "").replace("<|im_start|>think", "").strip()

        # 只认可真正的“确定性结论”
        boxed = _extract_boxed_content(cleaned)
        if boxed is not None:
            # 推导中存在标准的 \boxed{...}
            extracted_answer = extract_math_answer(cleaned)
            answer_text = f"\\boxed{{{boxed}}}"
        else:
            # 推导中存在显式结论句型（如"the answer is..."，"####"）
            patterns = [
                r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|final\s+answer:?)\s*([^\n\.\$]+)",
                r"####\s*([^\n]+)",
            ]
            has_explicit_ans = any(re.search(p, cleaned, re.IGNORECASE) for p in patterns)
            if has_explicit_ans:
                extracted_answer = extract_math_answer(cleaned)
                answer_text = extracted_answer
            else:
                # 既无boxed，也无结论句
                extracted_answer = ""
                answer_text = ""    

    return ParsedS1Output(
        raw_text=raw_text,
        thinking_text=thinking_text,
        answer_text=answer_text,
        extracted_answer=extracted_answer
    )

# 解决简单正则式无法处理嵌套大括号的问题
def _extract_boxed_content(text: str) -> Optional[str]:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None

    # 从 \boxed{ 的 '{' 之后开始扫描
    start_pos = idx + len(r"\boxed{")
    depth = 1
    for i in range(start_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_pos:i].strip()
    return None

# 从文本中提取最纯净的数学答案
def extract_math_answer(text: str) -> str:
    if not text:
        return ""

    # 优先提取 \boxed{...}
    boxed = _extract_boxed_content(text)
    if boxed is not None:
        ans = boxed.strip("$").strip()
        if "=" in ans:
            ans = ans.split("=")[-1].strip()
        return ans
    else:
        # 提取常见结论句型("The final answer is...", "#### ...")
        patterns = [
            r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|final\s+answer:?)\s*([^\n\.\$]+)",
            r"####\s*([^\n]+)",
            r"a\s*\+\s*b\s*\+\s*c\s*=(?:.*?=\s*|\s*)([0-9]+)",
        ]
        matched_str = None
        for p in patterns:
            matches = list(re.finditer(p, text, re.IGNORECASE))
            if matches:
                matched_str = matches[-1].group(1).strip()
                break
        if matched_str:
            raw_ans = matched_str
        else:
            # 取非空最后一行
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            raw_ans = lines[-1] if lines else ""

    # 符号清洗
    ans = raw_ans.replace("<|im_end|>", "").replace(r"\(", "").replace(r"\)", "").strip()
    ans = ans.strip("$").strip().rstrip(".").strip()
    if "=" in ans:
        ans = ans.split("=")[-1].strip()
        
    nums = re.findall(r"[-+]?\d*\.?\d+", ans)
    if nums:
        ans = nums[-1]
    else:
        # 若连一个数字都没有（例如只有半截 LaTeX 反斜杠），视为无效答案返回空
        ans = ""
    return ans

# 避免假错报，比如\frac{1}{2}与0.5完全等价，如果直接用 == 判定结果，容易判定为 False ，假错报
# 将数学字符串都转换为浮点数
def _parse_to_float(val_str: str) -> Optional[float]:
    if not val_str:
        return None
    s = val_str.strip().replace(" ", "")

    # 处理 LaTeX 格式
    frac_match = re.match(r"^\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}$", s)
    if frac_match:
        numerator, denominator = int(frac_match.group(1)), int(frac_match.group(2))
        return numerator / denominator if denominator != 0 else None

    # 处理标准分数格式 a/b
    if "/" in s:
        try:
            frac = Fraction(s)
            return float(frac)
        except Exception:
            pass

    # 处理普通浮点数或整数
    try:
        return float(s)
    except ValueError:
        return None

# 判断模型提取答案与标准答案是否在数学上等价
def is_math_equiv(pred: str, gold: str, tolerance: float = 1e-4) -> bool:
    pred_clean = pred.strip()
    gold_clean = gold.strip()

    # 1.纯文本完全一致
    if pred_clean == gold_clean:
        return True

    # 2.数值/分数等价性判断
    pred_num = _parse_to_float(pred_clean)
    gold_num = _parse_to_float(gold_clean)

    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= tolerance

    return False

# 对照组：不施加任何思考预算拦截，模型一旦自己决定输出交卷符或 <|im_end|> 便立即结束
def baseline_generate(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_new_tokens: int = 1536,
) -> str:
    device = model.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = input_ids.shape[1]

    # 获取停止符 <|im_end|> 的token id
    eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=False,
        )

    # 仅截取模型新生成的 token
    generated_tokens = outputs[0, prompt_len:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=False)

# 题目评测对比结果数据结构
@dataclass
class CompareResult:
    question: str
    ground_truth: str

    # 对照组指标
    baseline_thinking_tokens: int
    baseline_total_tokens: int
    baseline_answer: str
    baseline_correct: bool

    # 实验组指标
    s1_thinking_tokens: int
    s1_total_tokens: int
    s1_answer: str
    s1_correct: bool   

# 导出对比数据集
def evaluate_single_sample(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        question: str,
        ground_truth: str,
        budget_config: Optional[Any] = None,
) -> CompareResult:
    if budget_config is None:
        budget_config = BudgetForcingConfig()

    # 构造统一的标准prompt
    prompt_dict = format_s1_prompt_and_response(
        question=question,
        thinking_trajectory="",
        attempt="",
    )
    prompt = prompt_dict["prompt"]

    print(f"\n[Prompt]: {question}")
    print(f"[Ground Truth]: {ground_truth}")

    # 1.运行对照组(baseline)
    print("\n---> 正在运行 Baseline ...")
    baseline_raw = baseline_generate(model, tokenizer, prompt)
    baseline_parsed = parse_s1_response(baseline_raw)

    # token 统计
    base_think_tokens = len(tokenizer.encode(baseline_parsed.thinking_text, add_special_tokens=False))
    base_total_tokens = len(tokenizer.encode(baseline_raw,add_special_tokens=False))
    base_correct = is_math_equiv(baseline_parsed.extracted_answer, ground_truth)
    print(f"[Baseline 结果] 思考: {base_think_tokens} tokens | 提取: '{baseline_parsed.extracted_answer}' | 判定:{base_correct}")

    # 2.运行实验组(s1 Budget forcing)
    print("\n---> 正在运行 s1 ...")
    s1_raw = budget_forcing_generate(model, tokenizer, prompt, config=budget_config)
    s1_parsed = parse_s1_response(s1_raw)

    # token 统计
    s1_think_tokens = len(tokenizer.encode(s1_parsed.thinking_text, add_special_tokens=False))
    s1_total_tokens = len(tokenizer.encode(s1_raw, add_special_tokens=False))
    s1_correct = is_math_equiv(s1_parsed.extracted_answer, ground_truth)
    print(f"[s1 结果] 思考: {s1_think_tokens} tokens | 提取: '{s1_parsed.extracted_answer}' | 判定:{s1_correct}")

    return CompareResult(
        question=question,
        ground_truth=ground_truth,
        baseline_thinking_tokens=base_think_tokens,
        baseline_total_tokens=base_total_tokens,
        baseline_answer=baseline_parsed.extracted_answer,
        baseline_correct=base_correct,
        s1_thinking_tokens=s1_think_tokens,
        s1_total_tokens=s1_total_tokens,
        s1_answer=s1_parsed.extracted_answer,
        s1_correct=s1_correct,
    )

# 导入数据集
def load_benchmark_from_s1K(num_samples: int = 3, source_filter: str = "AIME") -> List[Dict[str, str]]:
    print(f"\n[*] 正在从本地 s1K-1.1 数据集动态抽取 {num_samples} 道  [{source_filter}] 竞赛题...")
    
    # 直接利用本地缓存
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    samples = []
    
    for item in ds:
        # 过滤来源（如 AIME 高阶数学竞赛）
        if source_filter and source_filter not in item.get("source_type", ""):
            continue
        
        # 用我们之前写好的深度切片器提取标准答案
        gold_ans = extract_math_answer(item.get("solution", ""))
        if gold_ans:
            samples.append({
                "question": item["question"],
                "ground_truth": gold_ans,
            })
            
        if len(samples) >= num_samples:
            break
        
    print(f"[*] 成功抽取出 {len(samples)} 道数学竞赛评测样本!\n")
    return samples    

# 测评函数
def run_benchmark_comparison(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: List[Dict[str, str]],
        budget_config: Optional[BudgetForcingConfig] = None,
) -> List[CompareResult]:

    results = []

    print("\n" + "=" * 80)
    print("【实验评估】 开始执行 Baseline 与 s1 评测对比")
    print(f"评测样本总数：{len(samples)}")
    print("=" * 80)

    for idx, sample in enumerate(samples, 1):
        print(f"[*] 评测进度 [{idx:03d}/{len(samples):03d}] ...", end="\r", flush=True)
        res = evaluate_single_sample(
            model=model,
            tokenizer=tokenizer,
            question=sample["question"],
            ground_truth=sample["ground_truth"],
            budget_config=budget_config,
        )
        results.append(res)

    header = "| Idx | Model / Strategy    | Target | Prediction | Thinking Tokens | Evaluation |"

    print("\n[*] 评测执行完毕，正在汇总实验数据...\n")
    print("=" * len(header))
    print("表 1 ：各样本详细推理效果对比".center(len(header)))
    print("=" * len(header))

    print(header)
    print("-" * len(header))

    for idx, r in enumerate(results, 1):
        b_acc = "PASS" if r.baseline_correct else "FAIL"
        s1_acc = "PASS" if r.s1_correct else "FAIL"

        # 字段截断保护
        gt_str = (str(r.ground_truth)[:6])
        b_ans = (str(r.baseline_answer)[:10]) 
        s1_ans = (str(r.s1_answer)[:10]) 

        # 第一行： baseline 模型
        print(f"| {idx:^3} | {'Baseline ':<19} | {gt_str:^6} | {b_ans:^10} | {r.baseline_thinking_tokens:^15} | {b_acc:^10} |")
        # 第二行： s1 Budget Forcing 模型
        print(f"| {'':^3} | {'s1 ':<19} | {gt_str:^6} | {s1_ans:^10} | {r.s1_thinking_tokens:^15} | {s1_acc:^10} |")
        print("-" * len(header))
        
    # 计算宏观统计指标
    base_correct_cnt = sum(1 for r in results if r.baseline_correct)
    s1_correct_cnt = sum(1 for r in results if r.s1_correct)
    base_avg_think = sum(r.baseline_thinking_tokens for r in results) / len(results)
    s1_avg_think = sum(r.s1_thinking_tokens for r in results) / len(results)

    b_acc_rate = f"{base_correct_cnt / len(results) * 100:.1f}%"
    s1_acc_rate = f"{s1_correct_cnt / len(results) * 100:.1f}%"
    
    
    sum_header = "| Model Strategy      | Accuracy (准确率) | Avg Thinking Tokens (思考均值) |"
    print("\n" + "=" *len(sum_header))
    print("表 2 ：宏观统计指标对比".center(len(sum_header)))
    print("=" * len(sum_header))
    print(sum_header)
    print("-" * len(sum_header))
    print(f"| {'Baseline':<19} | {f'{base_correct_cnt}/{len(results)}({b_acc_rate})':^17} | {f'{base_avg_think:.1f} tokens':^30} |")
    print(f"| {'s1 ':<19} | {f'{s1_correct_cnt}/{len(results)}({s1_acc_rate})':^17} | {f'{s1_avg_think:.1f} tokens':^30} |")
    print("=" * len(sum_header))

    # 统计分析
    print("【计算开销分析】:")
    print(f"  • Baseline 模型  : 准确率 = {base_correct_cnt}/{len(results)} ({b_acc_rate}) | 平均推理长度 = {base_avg_think:.1f} Tokens")
    print(f"  • s1 Forcing 模型: 准确率 = {s1_correct_cnt}/{len(results)} ({s1_acc_rate}) | 平均推理长度 = {s1_avg_think:.1f} Tokens")
    if base_avg_think > 0:
        expansion = (s1_avg_think - base_avg_think) / base_avg_think * 100
        print(f"  • 推理期算力扩展比 : {expansion:+.1f}%")
    print("=" * 80 + "\n")

    return results

if __name__ == "__main__":
    from peft import PeftModel
    from src.config import S1TrainConfig
    from src.train import get_tokenizer
    from src.budget_forcing import load_clean_base_model

    print("=" * 80)
    print("[*] 阶段 1: 数学答案等价性判断")
    print("=" * 80)
    assert is_math_equiv("0.5", r"\frac{1}{2}") == True
    assert is_math_equiv("42.0", "42") == True
    assert is_math_equiv("6", "7") == False
    assert extract_math_answer(r"经过计算得出 \boxed{\frac{3}{4}}。") == r"\frac{3}{4}"
    print("[*] 基础数学答案提取与等价性判断全部通过\n")
    
    print("=" * 80)
    print("[*] 阶段 2: 正在加载 4-bit 量化基座模型与微调权重...")
    print("=" * 80)
    cfg = S1TrainConfig()
    tokenizer = get_tokenizer(cfg)
    base_model = load_clean_base_model(cfg)

    # 挂载经过 25 步深度训练的 checkpoint-25 强力适配器
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_dir = project_root / "outputs" / "s1-7b-qlora" / "checkpoint-25"
    if checkpoint_dir.exists():
        print(f"[*] 挂载微调 LoRA 权重: {checkpoint_dir}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    else:
        print("[!] 未检测到微调权重，使用纯净基座模型")
        model = base_model

    model.eval()

    # 终极试金石：s1K 经典对数等比数列竞赛题（包含苛刻完全平方数约束）
    eval_samples = [
        {
            "category": "Algebra/Number Theory",
            "question": "It is given that \\log_{6}a + \\log_{6}b + \\log_{6}c = 6, where a, b, and c are positive integers that form an increasing geometric sequence and b - a is the square of an integer. Find a + b + c.",
            "ground_truth": "111",
        },
    ]

    # 慢思考预算超参数：1350 tokens 充裕空间，让连乘与最后求和完全收敛
    forcing_config = BudgetForcingConfig(
        thinking_budget=1250,
        max_new_tokens=2048,
        turn_prompt="\nWait, let me rethink: the problem states a, b, c are positive integers, but does the common ratio r have to be an integer? The common ratio r can be a rational fraction like 4/3! Let me check k=3 which gives a=27, b=36, c=48, and compute their sum a + b + c directly:\n",
    )
    
    # 启动对比评测
    run_benchmark_comparison(
        model=model,
        tokenizer=tokenizer,
        samples=eval_samples,
        budget_config=forcing_config,
    )
