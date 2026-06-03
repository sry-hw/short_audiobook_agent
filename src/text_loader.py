"""读取并清理文本文件，统一段落分隔格式。"""

import re
from pathlib import Path


def load_text(file_path: str) -> str:
    """读取 UTF-8 文本文件，统一为 \\n\\n 段落分隔格式。

    处理两种原始格式：
    - 空行分段（sample_story_01.txt）：原样保留
    - 缩进分段（不懂就要问.txt）：缩进换行转空行，行内换行合并

    Args:
        file_path: 文本文件路径

    Returns:
        用 \\n\\n 分段的统一格式文本

    Raises:
        FileNotFoundError: 文件不存在时抛出，附带文件路径信息
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path.resolve()}")

    text = path.read_text(encoding="utf-8")
    return _normalize(text)


def _normalize(text: str) -> str:
    """将文本统一为 \\n\\n 段落分隔格式。

    处理规则：
    - 空行 → 结束当前段落
    - 2+ 空格/制表符缩进 → 结束当前段落，开始新段落
    - 无缩进非空行 → 续行，拼到当前段落
    """
    lines = text.split("\n")
    paragraphs = []
    current = ""

    for line in lines:
        stripped = line.lstrip(" \t")
        indent_len = len(line) - len(stripped)

        if indent_len >= 2:
            # 缩进行 → 保存当前段落，开始新段落
            if current:
                paragraphs.append(current)
            current = stripped
        elif stripped:
            # 非空无缩进 → 拼到当前段落
            current += stripped
        else:
            # 空行 → 保存当前段落
            if current:
                paragraphs.append(current)
                current = ""

    if current:
        paragraphs.append(current)

    return "\n\n".join(paragraphs)
