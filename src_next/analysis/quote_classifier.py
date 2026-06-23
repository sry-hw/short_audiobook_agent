"""src_next/analysis/quote_classifier.py

引号内容语义分类：判断每个 dialogue candidate 是不是真的对白。

数据流位置：
    Segment[]（来自 core.segment_builder，所有引号都先切成 dialogue 候选）
        → classify_and_merge_quotes(segments, llm_client)
        → Segment[]（非对白引号已并回 narration，长度可能变短）

为什么需要这一步：
    core/segment_builder 用纯 regex 把所有中文引号内容都切成 dialogue 候选，
    不区分真对白和强调词 / 书名 / 术语。如果直接交给 resolve_speakers，
    "摇"、"摇花乐"、"西游记" 这种引号会被当成对白，下游 director_plan
    会按 dialogue 风格给情绪指导（不对），TTS 也会按对白段切开（多余）。

    旧 src/ 把这一步合在 llm_story_resolver 里（quote_type + speaker 一起问）。
    src_next/ 拆成两步：
        1. quote_classifier：判断 quote_type（是不是对白）
        2. story_resolver：对真正的对白判断 speaker
    两个 LLM prompt 各自专注，互不耦合。

契约：
    本函数允许 N → M（M ≤ N）：非对白引号并回相邻 narration，segment 总数变少。
    下游 resolve_speakers 是 1:1，不能改变 segment 数量。

只用 LLM 判断：
    quote_type 完全交给 LLM，不做规则 + LLM 混合判断。
    LLM 失败 / 结构异常 → 所有 candidate fallback 成 unknown → 按规则并回 narration。
"""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..core.data_models import Segment
from ..llm.base import BaseLLMClient, LLMError


# ── 常量 ────────────────────────────────────────────────────────────────────

# quote_type 取值
_KEEP_TYPES = ("dialogue", "inner_thought")  # 保留为独立 segment
_MERGE_TYPES = ("quoted_term", "title_or_name", "unknown")  # 并回 narration
_ALL_TYPES = _KEEP_TYPES + _MERGE_TYPES

# 合并时给原 dialogue 文本加回的引号（统一用智能双引号，与 core/segment_builder 一致）
_OPEN_QUOTE = "“"  # "
_CLOSE_QUOTE = "”"  # "


# ── LLM prompt ─────────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM_PROMPT = """你是一个中文故事文本分析师。给定一个段落原文和其中已经被切出来的若干引号内容（dialogue candidate），判断每个引号内容的语义类型。

严格输出 JSON，结构如下：

{
  "quote_classifications": [
    {
      "segment_id": "seg_003",
      "quote_type": "title_or_name",
      "reason": "该引号内容是书名",
      "confidence": 0.91
    }
  ]
}

quote_type 必须是以下 5 个值之一：

- dialogue      真实人物/角色说出的对白（包括大声喊叫、问问题等）
- inner_thought 心理活动（心想、暗想、暗自思忖、脑子里浮现的话）
- quoted_term   强调词、概念、术语、特殊称谓（不是角色说话）
- title_or_name 书名、文章名、章节标题、名称、标语等（不是角色说话）
- unknown       无法判断

判断依据：
- 引号前出现 "X 说/问/答/道/喊/叫/笑着说/哭着说" 等明显归属 → dialogue
- 引号前出现 "心想 / 暗想 / 暗自思忖 / 在心里说" → inner_thought
- 引号内容是单个字 / 短词，被作者用引号包裹以强调（如 "摇"、"不错"）→ quoted_term
- 引号内容是书名 / 文章名 / 章节名 / 标语（如 "西游记"、"咏鹅"）→ title_or_name
- 实在判断不出 → unknown（保守并回 narration）

要求：
- quote_classifications 数量必须等于输入 candidate 数量
- 每个输入 segment_id 都必须出现且唯一
- quote_type 必须是上面 5 个值之一（区分大小写，用下划线）
- reason 简短一句话即可
- confidence 是 0.0~1.0 的浮点数
"""


# ── 入口函数 ────────────────────────────────────────────────────────────────

def classify_and_merge_quotes(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
    output_debug_path: str | None = None,
) -> list[Segment]:
    """对 dialogue candidate 调 LLM 判 quote_type，非对白引号并回 narration。

    Args:
        segments: ``core.segment_builder.build_segments()`` 输出的 Segment 列表。
            其中所有引号内容都已切成 ``segment_type="dialogue", speaker="unknown"``
            的候选段。
        llm_client: 实现 ``BaseLLMClient`` 的任意后端。
        story_context: 故事标题 / 章节信息，附加给 LLM 提示。
        output_debug_path: 如果不为 None，把每个 candidate 的 LLM 判断结果
            保存成 ``quote_classifications.json`` 到这个路径。

    Returns:
        新的 Segment 列表（**长度可能 ≤ 输入**）。非对白引号已合并到相邻
        narration 段中（加回引号字符），dialogue/inner_thought 保留为独立
        segment。所有 segment_id 重新编号为 seg_001, seg_002, ...
    """
    if not segments:
        return []

    # 第一遍：调 LLM 拿 quote_type（按段落分组，每组一次调用）
    classifications = _classify_via_llm(segments, llm_client, story_context)
    id_to_type = {c["segment_id"]: c["quote_type"] for c in classifications}
    id_to_meta = {c["segment_id"]: c for c in classifications}

    # 第二遍：按规则合并 segment
    merged = _merge_segments(segments, id_to_type)

    # 第三遍：重编号 segment_id
    for i, seg in enumerate(merged, start=1):
        seg.segment_id = f"seg_{i:03d}"

    # 可选：debug 输出
    if output_debug_path is not None:
        _write_debug_json(segments, id_to_meta, classifications, output_debug_path)

    return merged


# ── LLM 调用 ────────────────────────────────────────────────────────────────

def _classify_via_llm(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    story_context: str,
) -> list[dict[str, Any]]:
    """按段落分组调 LLM，返回所有 candidate 的分类结果。

    失败 / 异常 → 返回空列表（让上层走 unknown 兜底）。
    """
    candidates_by_para = _collect_candidates_by_paragraph(segments)
    if not candidates_by_para:
        return []

    all_classifications: list[dict[str, Any]] = []

    for raw_index in sorted(candidates_by_para.keys()):
        group_segments = _segments_in_paragraph(segments, raw_index)
        candidates = candidates_by_para[raw_index]

        prompt = _build_classifier_prompt(group_segments, candidates, story_context)
        try:
            result = llm_client.generate_json(
                prompt, system_prompt=_CLASSIFIER_SYSTEM_PROMPT
            )
        except LLMError:
            result = None
        except Exception:  # noqa: BLE001
            result = None

        if result is None:
            # LLM 失败：本段所有 candidate fallback 成 unknown
            for c in candidates:
                all_classifications.append(_fallback_classification(c))
            continue

        parsed = _extract_classifications(result, candidates)
        all_classifications.extend(parsed)

    return all_classifications


def _collect_candidates_by_paragraph(
    segments: list[Segment],
) -> dict[int, list[Segment]]:
    """收集所有 dialogue candidate（segment_type=dialogue AND speaker=unknown），
    按 raw_index 分组。"""
    groups: dict[int, list[Segment]] = defaultdict(list)
    for seg in segments:
        if seg.segment_type == "dialogue" and seg.speaker == "unknown":
            groups[seg.raw_index].append(seg)
    return groups


def _segments_in_paragraph(
    segments: list[Segment], raw_index: int
) -> list[Segment]:
    """返回同一段落（raw_index 相同）的全部 segment，按原顺序。"""
    return [s for s in segments if s.raw_index == raw_index]


def _build_classifier_prompt(
    group_segments: list[Segment],
    candidates: list[Segment],
    story_context: str,
) -> str:
    parts: list[str] = []
    if story_context:
        parts.append(f"## 故事上下文\n\n{story_context}")

    # 用组内全部 segment 按原顺序拼回段落原文（dialogue 段加回引号）
    parts.append("\n## 段落原文（已按引号切分标记）\n")
    parts.append(_reconstruct_paragraph(group_segments))

    parts.append("\n## 待判断的引号内容\n")
    for seg in candidates:
        parts.append(f"- segment_id={seg.segment_id}\n  text={seg.text}")

    parts.append(
        "\n请对以上每个引号内容输出 quote_type，"
        "quote_classifications 数量必须等于输入 candidate 数量，"
        "segment_id 必须一一对应。"
    )
    return "\n".join(parts)


def _reconstruct_paragraph(group_segments: list[Segment]) -> str:
    """把同一段落切出来的 segments 按原顺序拼回近似段落原文。

    dialogue 段加回智能双引号，narration 段原样拼接。
    """
    pieces: list[str] = []
    for seg in group_segments:
        if seg.segment_type == "dialogue":
            pieces.append(_OPEN_QUOTE + seg.text + _CLOSE_QUOTE)
        else:
            pieces.append(seg.text)
    return "".join(pieces)


def _extract_classifications(
    result: Any,
    candidates: list[Segment],
) -> list[dict[str, Any]]:
    """从 LLM 返回里抠出 quote_classifications，并对齐到 candidate 列表。

    兼容多种返回形状；缺失 / 不合法的 candidate 自动补 fallback。
    """
    raw_list: list[dict[str, Any]] = []
    if isinstance(result, dict):
        items = result.get("quote_classifications")
        if isinstance(items, list):
            raw_list = [c for c in items if isinstance(c, dict)]
        elif "segment_id" in result:
            raw_list = [result]
    elif isinstance(result, list):
        raw_list = [c for c in result if isinstance(c, dict)]

    # 按 segment_id 索引
    id_to_raw: dict[str, dict[str, Any]] = {}
    for raw in raw_list:
        sid = str(raw.get("segment_id") or "").strip()
        if sid:
            id_to_raw[sid] = raw

    # 对齐到输入 candidate 顺序
    aligned: list[dict[str, Any]] = []
    for c in candidates:
        raw = id_to_raw.get(c.segment_id)
        if raw is None:
            aligned.append(_fallback_classification(c))
            continue
        quote_type = _clean_quote_type(raw.get("quote_type"))
        aligned.append({
            "segment_id": c.segment_id,
            "text": c.text,
            "quote_type": quote_type,
            "source": "llm",
            "reason": _clean_reason(raw.get("reason")),
            "confidence": _clean_confidence(raw.get("confidence")),
        })

    return aligned


def _fallback_classification(seg: Segment) -> dict[str, Any]:
    """LLM 失败时的兜底：保守按 unknown（会被 merge 回 narration）。"""
    return {
        "segment_id": seg.segment_id,
        "text": seg.text,
        "quote_type": "unknown",
        "source": "fallback",
        "reason": "LLM 调用失败或返回结构异常",
        "confidence": 0.0,
    }


# ── 合并逻辑 ────────────────────────────────────────────────────────────────

def _merge_segments(
    segments: list[Segment],
    id_to_type: dict[str, str],
) -> list[Segment]:
    """按 quote_type 决定保留 / 合并，输出新的 segment 列表（未重编号）。

    - dialogue / inner_thought → 保留为独立 segment（segment_type 改成 quote_type）
    - quoted_term / title_or_name / unknown → 加引号并回相邻 narration

    合并规则：在同一 raw_index 内累积 narration buffer，遇到合并型 candidate
    就把 "..." 加到 buffer，遇到保留型 candidate 就先 flush buffer 再 emit
    candidate。跨段落时强制 flush。
    """
    result: list[Segment] = []
    narration_buffer: list[str] = []
    current_raw_index: int | None = None

    def flush_narration(raw_index: int) -> None:
        if narration_buffer:
            text = "".join(narration_buffer).strip()
            if text:
                result.append(
                    Segment(
                        segment_id="",  # 占位，后面统一重编号
                        text=text,
                        speaker="narrator",
                        segment_type="narration",
                        raw_index=raw_index,
                    )
                )
        narration_buffer.clear()

    for seg in segments:
        # 跨段落 → 先 flush 上一段的 narration
        if current_raw_index is None:
            current_raw_index = seg.raw_index
        elif seg.raw_index != current_raw_index:
            flush_narration(current_raw_index)
            current_raw_index = seg.raw_index

        is_candidate = (
            seg.segment_type == "dialogue" and seg.speaker == "unknown"
        )
        if not is_candidate:
            # narration 段，直接累积
            narration_buffer.append(seg.text)
            continue

        quote_type = id_to_type.get(seg.segment_id, "unknown")
        if quote_type in _KEEP_TYPES:
            # 保留独立 segment：先 flush 已累积的 narration，再 emit candidate
            flush_narration(seg.raw_index)
            result.append(
                Segment(
                    segment_id="",
                    text=seg.text,
                    speaker="unknown",  # 交给 resolve_speakers 填
                    segment_type=quote_type,  # dialogue 或 inner_thought
                    raw_index=seg.raw_index,
                )
            )
        else:
            # 合并回 narration：加引号拼到 buffer
            narration_buffer.append(
                _OPEN_QUOTE + seg.text + _CLOSE_QUOTE
            )

    # 收尾 flush
    if current_raw_index is not None:
        flush_narration(current_raw_index)

    return result


# ── Debug 输出 ──────────────────────────────────────────────────────────────

def _write_debug_json(
    original_segments: list[Segment],
    id_to_meta: dict[str, dict[str, Any]],
    classifications: list[dict[str, Any]],
    output_path: str,
) -> None:
    """保存 quote_classifications.json 调试用。

    输出格式（按用户 spec）：

        {
          "version": 1,
          "items": [
            {
              "segment_id": "seg_003",      ← 这是 build_segments 阶段的原始 id
              "text": "摇",
              "quote_type": "quoted_term",
              "source": "llm" | "fallback",
              "reason": "...",
              "confidence": 0.0~1.0
            }
          ]
        }
    """
    # 用原始 segment_id（build_segments 阶段）作为锚点，避免合并后的重编号混淆
    items: list[dict[str, Any]] = []
    meta_by_id = {c["segment_id"]: c for c in classifications}
    for seg in original_segments:
        if not (seg.segment_type == "dialogue" and seg.speaker == "unknown"):
            continue
        meta = meta_by_id.get(seg.segment_id)
        if meta is None:
            meta = _fallback_classification(seg)
        items.append({
            "segment_id": seg.segment_id,
            "text": seg.text,
            "quote_type": meta.get("quote_type", "unknown"),
            "source": meta.get("source", "fallback"),
            "reason": meta.get("reason", ""),
            "confidence": meta.get("confidence", 0.0),
        })

    payload = {"version": 1, "items": items}

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── 字段清洗 ────────────────────────────────────────────────────────────────

def _clean_quote_type(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in _ALL_TYPES:
        return s
    return "unknown"


def _clean_reason(raw: Any) -> str:
    s = str(raw or "").strip()
    if len(s) > 120:
        s = s[:120]
    return s


def _clean_confidence(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
