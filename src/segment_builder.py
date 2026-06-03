"""Segment 构建器：合并 parser 和 resolver 结果，生成 TTS 可用的 segments。

每个 segment 只对应一个 speaker，是最小 TTS 单位。
"""

from typing import Dict, List, Tuple

# quoted_type 中需要合并回 narration 的类型
_MERGE_TYPES = {"quoted_term", "title_or_name", "unknown"}


def build_segments(parsed_result: Dict, resolved_result: Dict) -> Dict:
    """合并 parser 和 resolver 结果，生成 segments。

    Args:
        parsed_result: story_parser.parse_text() 的输出
        resolved_result: llm_story_resolver.resolve_quotes() 的输出

    Returns:
        包含 segments 列表的字典
    """
    resolution_map = _build_resolution_map(resolved_result)
    all_segments = []
    seg_id = 0

    for para in parsed_result["paragraphs"]:
        para_id = para["paragraph_id"]
        segments, seg_id = _build_for_paragraph(para, para_id, resolution_map, seg_id)
        all_segments.extend(segments)

    return {
        "segments": all_segments,
        "total_segments": len(all_segments),
    }


def _build_resolution_map(resolved_result: Dict) -> Dict[str, Dict]:
    """建立 part_id → resolution 的映射表。"""
    result = {}
    for group in resolved_result.get("resolutions", []):
        for r in group.get("quote_resolutions", []):
            result[r["part_id"]] = r
    return result


def _build_for_paragraph(
    para: Dict, para_id: int, resolution_map: Dict[str, Dict], seg_id: int
) -> Tuple[List[Dict], int]:
    """处理单个段落，返回 segments 和更新后的 seg_id。"""
    segments = []
    pending_narration = ""

    for part in para["parts"]:
        if part["type"] == "text":
            pending_narration += part["content"]
            continue

        # quoted part — 查 resolver 结果
        resolution = resolution_map.get(part["part_id"])
        quote_type = resolution.get("quote_type", "unknown") if resolution else "unknown"

        if quote_type in _MERGE_TYPES:
            # 非对话引号：带引号原文合并回 narration
            pending_narration += "“" + part["content"] + "”"
            continue

        # dialogue / inner_thought：先生出 pending narration，再出 dialogue
        if pending_narration.strip():
            seg_id += 1
            segments.append({
                "segment_id": seg_id,
                "paragraph_id": para_id,
                "type": "narration",
                "speaker": "narrator",
                "text": pending_narration.strip(),
            })
            pending_narration = ""

        seg_id += 1
        speaker = "unknown"
        if resolution:
            speaker = resolution.get("speaker", "unknown")
        segments.append({
            "segment_id": seg_id,
            "paragraph_id": para_id,
            "type": "dialogue",
            "speaker": speaker,
            "text": part["content"].strip(),
        })

    if pending_narration.strip():
        seg_id += 1
        segments.append({
            "segment_id": seg_id,
            "paragraph_id": para_id,
            "type": "narration",
            "speaker": "narrator",
            "text": pending_narration.strip(),
        })

    return segments, seg_id
