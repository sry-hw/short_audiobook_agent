"""src_next/core/segment_builder.py

文本切分：两级切分。

第一级 — 段落切分（按格式信号）：

1. 空行分段（如 input/sample_story_01.txt）：
   连续两个 \\n（中间夹空行）作为段落分隔。
2. 段首缩进分段（如 input/不懂就要问.txt）：
   行首有 2+ 空格/制表符缩进表示新段落开始。
3. 行内软换行续行：
   无缩进非空行视为上一段的续行，直接拼接到当前段落。

另外提供 **fallback**：如果结构化模式（空行 / 缩进）把全文压成 1 段、
但原文有多条非空行（典型例子：input/小红帽.txt 每行都是独立段落），
则回退到「每行一段」模式。

第二级 — 引号切分（每个段落内部）：

每个段落再按中文引号对进一步切分，产出交替的 narration / dialogue chunk。
- narration chunk：引号外的内容，speaker 默认 narrator
- dialogue chunk：引号内的内容（不含引号字符本身），speaker 默认 unknown
- 空引号（""）跳过，不产生 dialogue chunk
- 不匹配的引号（只有开头没有结尾）保持在 narration 里

支持两种引号对（与旧 src/story_parser.py 一致）：
- 智能双引号：" "
- 直角引号：「 」

切完之后**每个 segment 物理上只可能有一个 speaker**——结构上保证了
analysis/story_resolver 只需要为每个 dialogue segment 问一次 LLM。

segment_id 用独立计数器（seg_001, seg_002, ...），raw_index 是 0-indexed
的段落位置（同段落切出来的多个 segment 共享同一个 raw_index，方便
analysis 层按段落重组上下文）。

逻辑移植自：
- 旧 src/text_loader.py 的段落切分（_normalize）
- 旧 src/story_parser.py 的引号切分（_extract_parts）
区别在于：
- 这里产出 list[Segment] 而非归一化后的纯文本或 dict 结构；
- StoryInput.text 已经是读入的字符串，不再做文件 IO；
- 加了单行段落 fallback，处理 text_loader.py 未覆盖的纯单 \\n 格式；
- 把段落切和引号切合并在一个函数里，输出扁平的 Segment 列表。
"""

from .data_models import StoryInput, Segment


# ── 第一级：段落切分 ────────────────────────────────────────────────────────

def _split_structured(text: str) -> list[str]:
    """结构化切分：识别空行 / 段首缩进 / 续行三种信号（移植自 text_loader._normalize）。

    规则：
    - 空行 → 结束当前段落。
    - 行首 2+ 空格/制表符缩进 → 结束当前段落，开始新段落。
    - 无缩进非空行 → 视为续行，拼到当前段落末尾（不加空格，符合中文排版）。
    """
    lines = text.split("\n")
    paragraphs: list[str] = []
    current = ""

    for line in lines:
        stripped = line.lstrip(" \t")
        indent_len = len(line) - len(stripped)

        if indent_len >= 2:
            if current:
                paragraphs.append(current)
            current = stripped
        elif stripped:
            current += stripped
        else:
            if current:
                paragraphs.append(current)
                current = ""

    if current:
        paragraphs.append(current)

    return paragraphs


def _split_single_line(text: str) -> list[str]:
    """单行切分：每个非空行就是独立一段（fallback 模式）。

    用于没有任何空行 / 缩进信号、但每行都是独立段落的文本。
    """
    return [line.strip() for line in text.split("\n") if line.strip()]


def _split_into_paragraphs(text: str) -> list[str]:
    """先尝试结构化切分；若结果只有 1 段但原文有多行，回退到单行模式。

    这样可以同时覆盖：
    - 空行分段（sample_story_01.txt）
    - 段首缩进分段（不懂就要问.txt）
    - 纯单 \\n 分段（小红帽.txt）
    """
    paragraphs = _split_structured(text)

    non_empty_lines = [line for line in text.split("\n") if line.strip()]
    if len(paragraphs) <= 1 and len(non_empty_lines) > 1:
        # 结构化模式把多行文本压成 1 段，说明原文用单 \n 分段
        paragraphs = _split_single_line(text)

    return paragraphs


# ── 第二级：引号切分 ────────────────────────────────────────────────────────

# 与旧 src/story_parser.py 一致：只支持中文智能双引号和直角引号。
# 不支持 ASCII 双引号 / 单引号，避免和代码里的字符串定界符混淆。
_QUOTE_PAIRS: list[tuple[str, str]] = [
    ("“", "”"),  # “ ”  LEFT/RIGHT DOUBLE QUOTATION MARK
    ("「", "」"),  # 「 」
]


def _build_quote_regex() -> "tuple[str, str]":
    """构造引号匹配用的 open/close 字符类字符串。"""
    open_chars = "".join(re_escape(o) for o, _ in _QUOTE_PAIRS)
    close_chars = "".join(re_escape(c) for _, c in _QUOTE_PAIRS)
    return open_chars, close_chars


def re_escape(ch: str) -> str:
    """对单字符做 regex 转义。

    所有支持的引号都是 non-ASCII Unicode 字符，在 regex 字符类里不需要转义；
    但保留这个函数以防未来加入 ASCII 引号。
    """
    return ch


def _split_paragraph_by_quotes(text: str) -> list[tuple[str, str]]:
    """按引号边界把单段文本切成交替的 (chunk_type, chunk_text) 列表。

    chunk_type 取值：
    - "narration"：引号外的内容
    - "dialogue"：引号内的内容（不含引号字符本身）

    边界情况：
    - 空引号（""）→ 跳过，不产生 dialogue chunk
    - 不匹配的引号（开头存在但没有闭合）→ 保持在 narration 里
    - 完全没引号 → 返回 [("narration", text)]

    移植自旧 src/story_parser.py 的 _extract_parts，输出结构改成
    ``list[tuple[str, str]]`` 而非 ``list[tuple[str, str, dict]]``，
    因为 src_next 不再需要 part_id（segment_id 由 build_segments 全局分配）。
    """
    import re

    open_chars, close_chars = _build_quote_regex()
    parts: list[tuple[str, str]] = []
    pos = 0

    pattern = re.compile(
        f"[{open_chars}]([^{close_chars}]*)[{close_chars}]"
    )

    for m in pattern.finditer(text):
        before = text[pos:m.start()]
        if before:
            parts.append(("narration", before))

        dialogue = m.group(1)
        if dialogue:
            parts.append(("dialogue", dialogue))

        pos = m.end()

    tail = text[pos:]
    if tail:
        parts.append(("narration", tail))

    return parts or [("narration", text)]


# ── 入口：build_segments ────────────────────────────────────────────────────

def build_segments(story_input: StoryInput) -> list[Segment]:
    """按自然段 + 引号切分文本，生成 segments。

    自动识别三种原始段落格式 + fallback（详见模块 docstring）。
    段落切完之后再按引号对进一步切，每个引号内的内容独立成一个 dialogue
    segment，引号外的内容（可能被多个引号截断）成 narration segment。

    初始字段：
      * narration segment → speaker="narrator", segment_type="narration"
      * dialogue segment  → speaker="unknown",  segment_type="dialogue"
                            （交由 analysis/quote_classifier 判定 quote_type，
                             非对白引号会被合并回 narration；保留下来的对白
                             segment 交给 analysis/story_resolver 调 LLM 填 speaker）

    编号约定：
      * segment_id 用独立计数器：seg_001, seg_002, ...（1-indexed 显示）
      * raw_index 是 0-indexed 的段落位置：
        - 第一段 raw_index=0
        - 同一段切出来的多个 segment 共享同一个 raw_index

    raw_index 主要供 analysis/story_resolver 按段落重组上下文用
    （LLM 识别 speaker 需要看完整段落，不能只看孤立的引号内容）。
    """
    paragraphs = _split_into_paragraphs(story_input.text)

    segments: list[Segment] = []
    seg_counter = 0  # 用于 segment_id（1-indexed 显示，3 位补零）
    raw_index = 0    # 用于段落位置（0-indexed）

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        chunks = _split_paragraph_by_quotes(paragraph)
        for chunk_type, chunk_text in chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                # 空白 chunk（含被过滤的空引号）直接跳过
                continue

            seg_counter += 1
            if chunk_type == "dialogue":
                speaker = "unknown"
                segment_type = "dialogue"
            else:
                speaker = "narrator"
                segment_type = "narration"

            segments.append(
                Segment(
                    segment_id=f"seg_{seg_counter:03d}",
                    text=chunk_text,
                    speaker=speaker,
                    segment_type=segment_type,
                    raw_index=raw_index,
                )
            )

        raw_index += 1

    return segments
