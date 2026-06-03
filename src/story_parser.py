"""故事文本解析器：段落切分、引号提取、基础结构解析。

纯机械操作，不做任何语义判断。
说话人识别和引号类型判断交给 llm_story_resolver.py。
"""

import re
from typing import Dict, List, Tuple

_QUOTE_PAIRS = [("“", "”"), ("「", "」")]


def parse_text(text: str) -> Dict:
    """将文本拆分为段落，提取引号结构。

    Args:
        text: 已规范化的文本（由 text_loader 提供）

    Returns:
        包含段落列表和 parts 结构的字典
    """
    paragraphs = _split_paragraphs(text)
    result = []

    for i, para in enumerate(paragraphs):
        para_id = i + 1
        raw_parts = _extract_parts(para)

        parts = []
        for j, (ptype, content) in enumerate(raw_parts):
            parts.append(
                {
                    "part_id": f"p{para_id}_part{j + 1}",
                    "type": ptype,
                    "content": content,
                }
            )

        result.append(
            {
                "paragraph_id": para_id,
                "text": para,
                "has_quotes": any(p["type"] == "quoted" for p in parts),
                "parts": parts,
            }
        )

    return {
        "paragraphs": result,
        "total_paragraphs": len(result),
    }


def _split_paragraphs(text: str) -> List[str]:
    """按空行拆分段落。"""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _extract_parts(text: str) -> List[Tuple[str, str]]:
    """按引号边界拆分为交替的 (text, quoted) 片段列表。"""
    oq = "".join(re.escape(o) for o, _ in _QUOTE_PAIRS)
    cq = "".join(re.escape(c) for _, c in _QUOTE_PAIRS)

    parts: List[Tuple[str, str]] = []
    pos = 0

    for m in re.finditer(f"[{oq}]([^{cq}]*)[{cq}]", text):
        before = text[pos : m.start()]
        if before:
            parts.append(("text", before))

        dialogue = m.group(1)
        if dialogue:
            parts.append(("quoted", dialogue))

        pos = m.end()

    tail = text[pos:]
    if tail:
        parts.append(("text", tail))

    return parts or [("text", text)]
