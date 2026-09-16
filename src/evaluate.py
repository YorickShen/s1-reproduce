import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from fractions import Fraction

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer 

# 结构化模型输出结果
@dataclass
class ParsedS1Output:
    raw_text: str
    thinking_text: str
    answer_text: str
    extracted_answer: str

# 解析 s1/ChatML 输出格式
def parse_s1_response(raw_text: str) -> ParsedS1Output:
    think_pattern = r"<\|im_start\|>think\n(.*?)<\|im_start\|>answer\n"
    answer_pattern = r"<\|im_start\|>answer\n(.*?)(?:<\|im_end\|>|$)"

    # 提取思考部分
    think_match = re.search(think_pattern, raw_text, re.DOTALL)
    thinking_text = think_match.group(1).strip() if think_match else ""

    # 提取作答部分
    answer_match = re.search(answer_pattern, raw_text, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""

    if not thinking_text and not answer_text:
        answer_text = raw_text.strip()

    # 提取纯净答案
    target_for_extract = answer_text if answer_text else raw_text
    extracted_answer = extract_math_answer(target_for_extract)        

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
        raw_ans = boxed
    else:
        # 提取常见结论句型("The final answer is...", "#### ...")
        patterns = [
            r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|final\s+answer:?)\s*([^\n\.\$]+)",
            r"####\s*([^\n]+)",
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
    ans = raw_ans.strip()
    ans = ans.strip("$").strip()
    ans = ans.rstrip(".").strip()
    if ans.lower().startswith("x = ") or ans.lower().startswith("x="):
        ans = ans.split("=", 1)[-1].strip()

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

    