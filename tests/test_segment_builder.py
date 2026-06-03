"""segment_builder 模块测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from story_parser import parse_text
from segment_builder import build_segments


def _build(text, resolutions=None):
    """从文本和可选的 resolutions 构建 segments。"""
    parsed = parse_text(text)
    if resolutions is None:
        resolutions = {"resolutions": []}
    return build_segments(parsed, resolutions)


# ── 纯旁白段落 ────────────────────────────────────────────────


def test_narration_only():
    """无引号段落生成单个 narration segment。"""
    result = _build("这是一段没有引号的旁白。")
    segs = result["segments"]
    assert len(segs) == 1
    assert segs[0]["type"] == "narration"
    assert segs[0]["speaker"] == "narrator"
    assert segs[0]["text"] == "这是一段没有引号的旁白。"
    print("test_narration_only 通过")


# ── dialogue segment ─────────────────────────────────────────


def test_dialogue_segment():
    """dialogue quoted part 生成独立 segment，speaker 正确。"""
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 1,
                "paragraph_text": "",
                "quote_resolutions": [
                    {
                        "part_id": "p1_part2",
                        "quote_type": "dialogue",
                        "speaker": "小明",
                        "confidence": "high",
                        "reason": "",
                        "content": "你好！",
                    }
                ],
            }
        ]
    }
    result = _build('小明说："你好！"', resolutions)
    segs = result["segments"]
    assert len(segs) == 2
    assert segs[0]["type"] == "narration"
    assert segs[0]["text"] == "小明说："
    assert segs[1]["type"] == "dialogue"
    assert segs[1]["speaker"] == "小明"
    assert segs[1]["text"] == "你好！"
    print("test_dialogue_segment 通过")


# ── quoted_term 合并回 narration ──────────────────────────────


def test_quoted_term_merged():
    """quoted_term 不单独成 segment，带引号合并回 narration。"""
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 1,
                "paragraph_text": "",
                "quote_resolutions": [
                    {
                        "part_id": "p1_part2",
                        "quote_type": "quoted_term",
                        "speaker": None,
                        "confidence": "high",
                        "reason": "",
                        "content": "摇",
                    }
                ],
            }
        ]
    }
    result = _build('就应当"摇"。', resolutions)
    segs = result["segments"]
    assert len(segs) == 1
    assert segs[0]["type"] == "narration"
    assert '"摇"' in segs[0]["text"]
    assert segs[0]["text"] == '就应当"摇"。'
    print("test_quoted_term_merged 通过")


# ── segment_id 全局递增 ──────────────────────────────────────


def test_segment_id_incremental():
    """segment_id 从 1 开始全局递增，跨段落连续。"""
    text = "第一段。\n\n他说："走。"她问："去哪？""
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 2,
                "paragraph_text": "",
                "quote_resolutions": [
                    {
                        "part_id": "p2_part2",
                        "quote_type": "dialogue",
                        "speaker": "他",
                        "confidence": "high",
                        "reason": "",
                        "content": "走。",
                    },
                    {
                        "part_id": "p2_part4",
                        "quote_type": "dialogue",
                        "speaker": "她",
                        "confidence": "high",
                        "reason": "",
                        "content": "去哪？",
                    },
                ],
            }
        ]
    }
    result = _build(text, resolutions)
    segs = result["segments"]
    ids = [s["segment_id"] for s in segs]
    assert ids == [1, 2, 3, 4]
    print("test_segment_id_incremental 通过")


# ── 混合段落：旁白 + 对话 + 旁白 + 对话 ──────────────────────


def test_mixed_paragraph():
    """一个段落中旁白和对话交替出现，各生成独立 segment。"""
    text = '母亲担心："可别来台风！"母亲念着："只要不来就好。"'
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 1,
                "paragraph_text": "",
                "quote_resolutions": [
                    {
                        "part_id": "p1_part2",
                        "quote_type": "dialogue",
                        "speaker": "母亲",
                        "confidence": "high",
                        "reason": "",
                        "content": "可别来台风！",
                    },
                    {
                        "part_id": "p1_part4",
                        "quote_type": "dialogue",
                        "speaker": "母亲",
                        "confidence": "high",
                        "reason": "",
                        "content": "只要不来就好。",
                    },
                ],
            }
        ]
    }
    result = _build(text, resolutions)
    segs = result["segments"]
    assert len(segs) == 4
    assert segs[0]["type"] == "narration"
    assert segs[1]["type"] == "dialogue"
    assert segs[1]["speaker"] == "母亲"
    assert segs[2]["type"] == "narration"
    assert segs[3]["type"] == "dialogue"
    assert segs[3]["speaker"] == "母亲"
    print("test_mixed_paragraph 通过")


# ── unknown 类型也合并回 narration ───────────────────────────


def test_unknown_merged():
    """unknown quote_type 合并回 narration，保留引号。"""
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 1,
                "paragraph_text": "",
                "quote_resolutions": [
                    {
                        "part_id": "p1_part2",
                        "quote_type": "unknown",
                        "speaker": None,
                        "confidence": "low",
                        "reason": "",
                        "content": "某某",
                    }
                ],
            }
        ]
    }
    result = _build('他说的"某某"是什么意思。', resolutions)
    segs = result["segments"]
    assert len(segs) == 1
    assert segs[0]["type"] == "narration"
    assert '"某某"' in segs[0]["text"]
    print("test_unknown_merged 通过")


# ── 无 resolution 的 quoted part 默认 unknown ─────────────────


def test_no_resolution_defaults_unknown():
    """quoted part 没有对应 resolution 时按 unknown 处理，合并回 narration。"""
    result = _build('这是"某个词"的用法。')
    segs = result["segments"]
    assert len(segs) == 1
    assert segs[0]["type"] == "narration"
    assert '"某个词"' in segs[0]["text"]
    print("test_no_resolution_defaults_unknown 通过")


# ── 桂花雨完整测试 ───────────────────────────────────────────


def test_guihua():
    """桂花雨.txt 全流程：parsed + 手工 resolutions → segments 验证。"""
    from text_loader import load_text

    file_path = Path(__file__).resolve().parent.parent / "input" / "桂花雨.txt"
    text = load_text(str(file_path))
    parsed = parse_text(text)

    # 使用手工 resolutions（与 guihua_resolved3.json 一致）
    resolutions = {
        "resolutions": [
            {
                "paragraph_id": 3,
                "paragraph_text": "",
                "quote_resolutions": [
                    {"part_id": "p3_part2", "quote_type": "dialogue", "speaker": "母亲", "confidence": "high", "reason": "", "content": "可别来台风啊！"},
                    {"part_id": "p3_part4", "quote_type": "dialogue", "speaker": "母亲", "confidence": "high", "reason": "", "content": "只要不来台风，我就可以收几大箩。送一箩给胡家老爷爷，送一箩给毛家老婆婆，他们两家糕饼做得多。"},
                ],
            },
            {
                "paragraph_id": 4,
                "paragraph_text": "",
                "quote_resolutions": [
                    {"part_id": "p4_part2", "quote_type": "quoted_term", "speaker": None, "confidence": "high", "reason": "", "content": "摇"},
                ],
            },
            {
                "paragraph_id": 5,
                "paragraph_text": "",
                "quote_resolutions": [
                    {"part_id": "p5_part2", "quote_type": "dialogue", "speaker": "我", "confidence": "high", "reason": "", "content": "妈，怎么还不摇桂花呢？"},
                    {"part_id": "p5_part4", "quote_type": "dialogue", "speaker": "母亲", "confidence": "high", "reason": "", "content": "还早呢，花开的时间太短，摇不下来的。"},
                    {"part_id": "p5_part6", "quote_type": "dialogue", "speaker": "我", "confidence": "high", "reason": "", "content": "啊！真像下雨，好香的雨呀！"},
                ],
            },
            {
                "paragraph_id": 7,
                "paragraph_text": "",
                "quote_resolutions": [
                    {"part_id": "p7_part2", "quote_type": "dialogue", "speaker": "母亲", "confidence": "high", "reason": "", "content": "这里的桂花再香，也比不上家乡院子里的桂花。"},
                ],
            },
            {
                "paragraph_id": 8,
                "paragraph_text": "",
                "quote_resolutions": [
                    {"part_id": "p8_part2", "quote_type": "quoted_term", "speaker": None, "confidence": "high", "reason": "", "content": "摇花乐"},
                ],
            },
        ]
    }

    result = build_segments(parsed, resolutions)
    segs = result["segments"]

    # 总数 17
    assert result["total_segments"] == 17

    # "摇" 合并到 paragraph 4 的 narration
    p4_segs = [s for s in segs if s["paragraph_id"] == 4]
    assert len(p4_segs) == 1
    assert p4_segs[0]["type"] == "narration"
    assert '"摇"' in p4_segs[0]["text"]

    # "摇花乐" 合并到 paragraph 8 的 narration
    p8_segs = [s for s in segs if s["paragraph_id"] == 8]
    assert len(p8_segs) == 1
    assert p8_segs[0]["type"] == "narration"
    assert '"摇花乐"' in p8_segs[0]["text"]

    # 所有 dialogue 的 speaker
    dialogue_segs = [s for s in segs if s["type"] == "dialogue"]
    speakers = [s["speaker"] for s in dialogue_segs]
    assert speakers.count("母亲") == 4
    assert speakers.count("我") == 2
    assert len(dialogue_segs) == 6

    # segment_id 连续
    ids = [s["segment_id"] for s in segs]
    assert ids == list(range(1, 18))

    print("test_guihua 通过")


# ── 运行 ──────────────────────────────────────────────────────


if __name__ == "__main__":
    test_narration_only()
    test_dialogue_segment()
    test_quoted_term_merged()
    test_segment_id_incremental()
    test_mixed_paragraph()
    test_unknown_merged()
    test_no_resolution_defaults_unknown()
    test_guihua()
    print("\n全部测试通过")
