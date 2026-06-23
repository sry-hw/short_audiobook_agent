"""src_next/analysis/story_resolver.py

说话人识别（填 dialogue / inner_thought segment 的 speaker）。

数据流位置：
    Segment[]（来自 quote_classifier.classify_and_merge_quotes，
              非对白引号已经并回 narration）
        → resolve_speakers(segments, llm_client)
        → 同样长度的 Segment[]（dialogue / inner_thought 段的 speaker 已被 LLM 填充）

策略：
    核心假设：上游 quote_classifier 已经判定每个引号内容的语义类型，只把
    ``dialogue`` 和 ``inner_thought`` 类型保留为独立 segment（speaker=unknown）。
    本层只负责为这些 segment 问 LLM "这句话是谁说的（或谁在心里想）"。

    1. 把 segments 按 raw_index 分组（同组来自同一段落）。
    2. 对每组：
       - 收集组内所有 dialogue / inner_thought segment。
       - 如果组内没有 → 跳过。
       - 否则用组内全部 segment 按原顺序拼回近似段落原文
         （dialogue 段加回引号字符）作为上下文，连同候选列表一起丢 LLM。
    3. 把 LLM 返回的 speaker 写回对应 segment。
    4. LLM 失败 / 没覆盖到的 → speaker fallback 成 narrator。

契约：1:1 输入输出。``resolve_speakers`` 不允许新增或删除 Segment，
只改 dialogue / inner_thought segment 的 speaker 字段。

本文件不做的事：
- 不发 HTTP 请求；
- 不读 .env；
- 不 import QwenHTTPClient / Gemma4HTTPClient；
- 不做引号切分（core/segment_builder 负责）；
- 不判引号是不是对白（quote_classifier 负责）；
- 不改 segment 数量（1:1）。
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from ..core.data_models import Segment
from ..llm.base import BaseLLMClient, LLMError


# segment_type 取值：本层会处理的两种"需要问 speaker"的类型。
# narration 已经是 narrator，不需要问 LLM。
_RESOLVABLE_TYPES = ("dialogue", "inner_thought")


# ── LLM prompt ─────────────────────────────────────────────────────────────

_RESOLVER_SYSTEM_PROMPT = """你是一个中文故事对话分析师。给定一个段落原文和其中的若干对白 / 心理活动（已经按引号切好，并已确认是对白或心理活动），判断每条的说话人。

严格输出 JSON，结构如下：

{
  "resolutions": [
    {
      "segment_id": "seg_003",
      "speaker": "说话人名称",
      "confidence": 0.0~1.0
    }
  ]
}

判断依据：
- 说话人通常出现在引号前的 "X 说/问/答/道/喊/叫" 等模式里
- 心理活动（"心想/暗想"）的说话人是动作执行者本人
- 一条对白 / 心理活动只属于一个说话人
- 如果对白前没有明确归属线索，结合上一条对白 / 段落上下文判断（连续对话常常是两个角色轮流说）
- 实在判断不出 → speaker 填 "narrator"（让下游当作旁白处理）

要求：
- resolutions 数量必须等于输入的对白 / 心理活动数量
- 每个 segment_id 都必须出现且唯一
- speaker 必须是真实角色名（1~6 字），不要把状态副词（如 "气喘吁吁地"、"低声"、"慢慢地"）当作 speaker
- speaker 不要包含标点 / 引号
- confidence 是 0.0~1.0 的浮点数
"""


# speaker 清洗时需要从首尾 strip 掉的字符（空白 + 中英文标点 + 各类引号）。
_SPEAKER_EDGE_STRIP = (
    " \t\n\r"
    "：:,，。！？、；;."
    "“”‘’"  # “ ” ‘ ’
    "「」『』"  # 「 」 『 』
    "\'\""                       # ASCII single + double quote
)


# 段落上下文里给 dialogue 加回的引号对（让 LLM 看到的上下文接近原文）。
# 与 core/segment_builder._QUOTE_PAIRS 保持一致。
_QUOTE_WRAP_PAIRS = {
    "“": "”",
    "「": "」",
}
_DEFAULT_OPEN_QUOTE = "“"
_DEFAULT_CLOSE_QUOTE = "”"


# ── 入口函数 ────────────────────────────────────────────────────────────────

def resolve_speakers(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[Segment]:
    """为每个 dialogue / inner_thought segment 填充 speaker（1:1 输入输出）。

    Args:
        segments: 通常是 ``quote_classifier.classify_and_merge_quotes()`` 输出
            的 Segment 列表。其中 ``segment_type="dialogue"`` 或
            ``"inner_thought"`` 的 segment 的 speaker 应为 ``"unknown"``，
            等待本函数填充；``segment_type="narration"`` 的 speaker 已是
            ``"narrator"``。
        llm_client: 实现 ``BaseLLMClient`` 的任意后端（Mock / Qwen / Gemma4）。
        story_context: 故事标题 / 章节信息，附加给 LLM 提示。

    Returns:
        新的 Segment 列表（深拷贝，入参不变；**长度严格等于输入**）。
        每个原 dialogue / inner_thought segment 的 speaker 被填成 LLM 判定
        的角色名，失败时填成 narrator。narration segment 不动。
    """
    if not segments:
        return []

    # 深拷贝，不动入参；全程不增删 Segment。
    resolved: list[Segment] = [deepcopy(seg) for seg in segments]

    # 按 raw_index 分组（同段落切出来的多个 segment 共享同一个 raw_index）
    groups: dict[int, list[Segment]] = defaultdict(list)
    for seg in resolved:
        groups[seg.raw_index].append(seg)

    for raw_index in sorted(groups.keys()):
        group = groups[raw_index]
        candidates = [
            s for s in group if s.segment_type in _RESOLVABLE_TYPES
        ]
        if not candidates:
            # 整段都是 narration，narrator 已就位，不需要问 LLM
            continue

        resolutions = _resolve_paragraph_dialogues(
            group, candidates, llm_client, story_context
        )
        id_to_res = {r["segment_id"]: r for r in resolutions}

        for seg in candidates:
            res = id_to_res.get(seg.segment_id)
            if res:
                seg.speaker = _clean_speaker(res.get("speaker") or "narrator")
            else:
                # LLM 没覆盖到这条 → narrator 兜底
                seg.speaker = "narrator"

    # 二次兜底：所有仍为 unknown / 空 speaker 的段都改成 narrator。
    for seg in resolved:
        if not seg.speaker or seg.speaker == "unknown":
            seg.speaker = "narrator"

    return resolved


# ── LLM 调用 ────────────────────────────────────────────────────────────────

def _resolve_paragraph_dialogues(
    group: list[Segment],
    dialogue_segs: list[Segment],
    llm_client: BaseLLMClient,
    story_context: str,
) -> list[dict[str, Any]]:
    """对单个段落内的所有 dialogue segment 调一次 LLM。

    失败 / 结构异常 → 返回空列表（让上层走 narrator 兜底）。
    """
    prompt = _build_resolver_prompt(group, dialogue_segs, story_context)
    try:
        result = llm_client.generate_json(
            prompt, system_prompt=_RESOLVER_SYSTEM_PROMPT
        )
    except LLMError:
        return []
    except Exception:  # noqa: BLE001
        # 任何意外都不能让整条链路挂
        return []

    return _extract_resolutions(result)


def _build_resolver_prompt(
    group: list[Segment],
    dialogue_segs: list[Segment],
    story_context: str,
) -> str:
    """构造单段 LLM prompt：段落上下文 + 待判断的 dialogue 列表。"""
    parts: list[str] = []
    if story_context:
        parts.append(f"## 故事上下文\n\n{story_context}")

    # 用组内全部 segment 按原顺序拼回段落原文
    # （dialogue 段加回引号字符，让 LLM 看到的上下文接近原文）。
    parts.append("\n## 段落原文（已按引号切分标记）\n")
    parts.append(_reconstruct_paragraph(group))

    parts.append("\n## 待判断的对白\n")
    for seg in dialogue_segs:
        parts.append(f"- segment_id={seg.segment_id}\n  text={seg.text}")

    parts.append(
        "\n请对以上每条对白输出 speaker，"
        "resolutions 数量必须等于输入对白数，segment_id 必须一一对应。"
    )
    return "\n".join(parts)


def _reconstruct_paragraph(group: list[Segment]) -> str:
    """把同一段落切出来的 segments 按原顺序拼回近似段落原文。

    - narration segment：原样拼接
    - dialogue / inner_thought segment：前后加回引号字符

    这只是给 LLM 看上下文用的，不要求 100% 还原原文（标点细节可能丢失）。
    """
    pieces: list[str] = []
    for seg in group:
        if seg.segment_type in _RESOLVABLE_TYPES:
            pieces.append(_DEFAULT_OPEN_QUOTE + seg.text + _DEFAULT_CLOSE_QUOTE)
        else:
            pieces.append(seg.text)
    return "".join(pieces)


def _extract_resolutions(result: Any) -> list[dict[str, Any]]:
    """从 LLM 返回里抠出 resolutions 列表。

    兼容多种形状：
    * ``{"resolutions": [...]}``  ← 期望
    * ``[{segment_id, ...}, ...]``← 旧 list 形式
    * ``{"source": "mock", ...}`` ← MockLLMClient 默认 dict
    * 其他任何异常形状

    MockLLM / 异常形状 → 返回空列表（让上层走 narrator 兜底）。
    """
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]

    if isinstance(result, dict):
        resolutions = result.get("resolutions")
        if isinstance(resolutions, list):
            return [r for r in resolutions if isinstance(r, dict)]
        # dict 看起来像单条 resolution（有 segment_id / speaker 字段）
        if "segment_id" in result or "speaker" in result:
            return [result]

    return []


# ── 字段清洗 ────────────────────────────────────────────────────────────────

def _clean_speaker(raw: Any) -> str:
    """清洗 speaker 字符串：去首尾空白 / 标点 / 引号；超长截断到 8 字。"""
    if not raw:
        return ""
    s = str(raw).strip().strip(_SPEAKER_EDGE_STRIP)
    if len(s) > 8:
        s = s[:8]
    return s
