import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

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

    return ParsedS1Output(
        raw_text=raw_text,
        thinking_text=thinking_text,
        answer_text=answer_text,
        extracted_answer=""
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

