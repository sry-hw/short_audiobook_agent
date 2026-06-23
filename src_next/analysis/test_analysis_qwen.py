"""src_next/analysis/test_analysis_qwen.py

用真实 Qwen3.6-plus + 真实 Qwen3-TTS VoiceDesign 跑前 7 步的端到端 smoke 测试。

链路：
    input/<story>.txt
        → StoryInput
        → build_segments                              ← core.segment_builder
        → classify_and_merge_quotes(segments, llm)    ← analysis.quote_classifier
        → resolve_speakers(merged, llm)               ← analysis.story_resolver
        → analyze_characters(resolved, llm)           ← analysis.character_analyzer
        → prepare_voicebank(chars, voicebank_adapter) ← voicebank.qwen_voicegenerator
                                                       (真实 Qwen3-TTS VoiceDesign)
        → generate_director_plan(resolved, chars, llm)← analysis.story_director
        → build_tts_instructions(...)                 ← core.tts_instruction_builder
        → 7 个 JSON 文件 + voicebank/*.wav 落到 <output_dir>/<story_name>/

用法：
    python src_next/analysis/test_analysis_qwen.py
    python src_next/analysis/test_analysis_qwen.py input/桂花雨.txt
    python src_next/analysis/test_analysis_qwen.py input/桂花雨.txt \\
        --output output-src-next-analysis-test

或作为模块运行（Windows 路径有时更稳）：
    python -m src_next.analysis.test_analysis_qwen input/桂花雨.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# Windows GBK 终端兼容：把 stdout/stderr 切成 UTF-8，避免打印中文报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# 导入要在 import 阶段就把 UTF-8 wrapper 装好，所以业务 import 放在 wrapper 之后
from src_next.core.data_models import (  # noqa: E402
    CharacterProfile,
    DirectorInstruction,
    Segment,
    StoryInput,
    TTSInstruction,
    VoicebankResult,
)
from src_next.core.segment_builder import build_segments  # noqa: E402
from src_next.core.tts_instruction_builder import build_tts_instructions  # noqa: E402
from src_next.llm.qwen_http import QwenHTTPClient  # noqa: E402
from src_next.voicebank.registry import create_voicebank_adapter  # noqa: E402
from src_next.analysis.quote_classifier import classify_and_merge_quotes  # noqa: E402
from src_next.analysis.story_resolver import resolve_speakers  # noqa: E402
from src_next.analysis.character_analyzer import analyze_characters  # noqa: E402
from src_next.analysis.story_director import generate_director_plan  # noqa: E402


_DEFAULT_INPUT = "input/桂花雨.txt"
_DEFAULT_OUTPUT = "output-src-next-analysis-test"
_VOICEBANK_PROFILE = "src_next/profiles/blue_qwen_voicegenerator.yaml"


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="src_next analysis + voicebank 端到端 smoke test"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=_DEFAULT_INPUT,
        help=f"输入故事文本路径（默认: {_DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=_DEFAULT_OUTPUT,
        help=f"输出根目录（默认: {_DEFAULT_OUTPUT}）",
    )
    return parser.parse_args()


# ── 路径解析 ────────────────────────────────────────────────────────────────

def _story_name_from_input(input_path: Path) -> str:
    """story_name 用输入文件的 stem（不含扩展名）。"""
    return input_path.stem


def _resolve_story_root(output_root: str, story_name: str) -> Path:
    """故事级根目录：<output_root>/<story_name>/

    下面会分出 ``json/`` 和 ``voicebank/`` 两个子目录。
    """
    story_root = Path(output_root).expanduser().resolve() / story_name
    story_root.mkdir(parents=True, exist_ok=True)
    return story_root


def _resolve_json_dir(story_root: Path) -> Path:
    json_dir = story_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir


# ── JSON 序列化 ─────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _segments_to_dicts(segments: list[Segment]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(s) for s in segments]


def _characters_to_dicts(chars: list[CharacterProfile]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(c) for c in chars]


def _plan_to_dicts(plan: list[DirectorInstruction]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(d) for d in plan]


def _tts_instructions_to_dicts(
    instructions: list[TTSInstruction],
) -> list[dict[str, Any]]:
    return [dataclasses.asdict(i) for i in instructions]


def _voicebank_result_to_dict(result: VoicebankResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _read_story(path: Path) -> StoryInput:
    text = path.read_text(encoding="utf-8")
    return StoryInput(
        story_name=path.stem,
        text=text,
        source_path=str(path),
    )


def _count_by_type(segments: list[Segment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seg in segments:
        counts[seg.segment_type] = counts.get(seg.segment_type, 0) + 1
    return counts


# ── Voicebank ───────────────────────────────────────────────────────────────

def _load_voicebank_profile(profile_path: Path) -> tuple[str, dict[str, Any]]:
    """从 yaml profile 加载 voicebank 配置段。

    Returns:
        (backend, config_dict)
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    vb = profile.get("voicebank") or {}
    backend = vb.pop("backend")
    return backend, vb


def _prepare_voicebank(
    characters: list[CharacterProfile],
    story_root: Path,
    profile_path: Path,
) -> VoicebankResult:
    """调用真实 Qwen3-TTS VoiceDesign 为每个角色生成 voice reference。

    Args:
        characters: analyze_characters 输出（含 narrator）。
        story_root: 故事级输出根目录；voicebank wavs 会落到
            ``<story_root>/voicebank/``。
        profile_path: voicebank profile yaml 路径。

    Returns:
        VoicebankResult（speaker_to_voice 映射 + 实际 wav 路径）。
    """
    backend, vb_config = _load_voicebank_profile(profile_path)
    adapter = create_voicebank_adapter(backend, **vb_config)
    # adapter 会在 <story_root> 下创建 output_subdir（默认 "voicebank"）子目录
    return adapter.prepare_voicebank(characters, str(story_root))


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        return 2

    story_name = _story_name_from_input(input_path)
    story_root = _resolve_story_root(args.output, story_name)
    json_dir = _resolve_json_dir(story_root)

    print("=" * 60)
    print("[analysis + voicebank smoke] 配置")
    print(f"  input          = {input_path}")
    print(f"  story_name     = {story_name}")
    print(f"  story_root     = {story_root}")
    print(f"  json_dir       = {json_dir}")
    print(f"  voicebank yaml = {_VOICEBANK_PROFILE}")
    print("=" * 60)
    print()

    # 1. 读故事 + build_segments
    story = _read_story(input_path)
    t0 = time.time()
    segments_raw = build_segments(story)
    t_seg = time.time() - t0
    raw_counts = _count_by_type(segments_raw)
    print(f"[1/8] build_segments            → {len(segments_raw)} 段  "
          f"(types={raw_counts})  ({t_seg:.2f}s)")

    # 2. 构造 Qwen LLM client（构造可能抛 LLMError）
    print("[2/8] QwenHTTPClient ...")
    try:
        llm = QwenHTTPClient()
    except Exception as err:  # noqa: BLE001
        print(
            f"[ERROR] QwenHTTPClient 构造失败: {type(err).__name__}: {err}\n"
            "请检查 .env 中 QWEN_BASE_URL / QWEN_API_KEY / QWEN_MODEL 是否配置正确。",
            file=sys.stderr,
        )
        return 3
    print(f"      base_url = {llm.base_url}")
    print(f"      model    = {llm.model}")
    print()

    # 3. classify_and_merge_quotes
    t0 = time.time()
    debug_path = json_dir / "quote_classifications.json"
    segments_merged = classify_and_merge_quotes(
        segments_raw,
        llm,
        story_context=story.story_name,
        output_debug_path=str(debug_path),
    )
    t_classify = time.time() - t0
    merged_counts = _count_by_type(segments_merged)
    n_merged_back = len(segments_raw) - len(segments_merged)
    print(f"[3/8] classify_and_merge_quotes → {len(segments_merged)} 段  "
          f"(types={merged_counts}; 合并回 narration: {n_merged_back})  "
          f"({t_classify:.2f}s)")

    # 4. resolve_speakers
    t0 = time.time()
    resolved = resolve_speakers(segments_merged, llm, story_context=story.story_name)
    t_resolve = time.time() - t0
    resolved_counts = _count_by_type(resolved)
    speakers_in_dialogue = sorted({
        s.speaker for s in resolved
        if s.segment_type in ("dialogue", "inner_thought")
        and s.speaker and s.speaker != "narrator"
    })
    print(f"[4/8] resolve_speakers          → {len(resolved)} 段  "
          f"(types={resolved_counts}; dialogue/inner_thought speakers="
          f"{speakers_in_dialogue})  ({t_resolve:.2f}s)")

    # 5. analyze_characters
    t0 = time.time()
    characters = analyze_characters(resolved, llm, story_context=story.story_name)
    t_char = time.time() - t0
    print(f"[5/8] analyze_characters        → {len(characters)} 个角色  "
          f"(narrator at idx 0: {characters[0].name == 'narrator'})  "
          f"({t_char:.2f}s)")

    # 6. prepare_voicebank（真实 Qwen3-TTS VoiceDesign）
    t0 = time.time()
    profile_path = Path(_VOICEBANK_PROFILE).expanduser().resolve()
    if not profile_path.exists():
        print(f"[ERROR] voicebank profile 不存在: {profile_path}", file=sys.stderr)
        return 4
    print(f"[6/8] prepare_voicebank ... (Qwen3-TTS VoiceDesign；每个角色 ~40-60s)")
    voicebank_result = _prepare_voicebank(characters, story_root, profile_path)
    t_vb = time.time() - t0
    n_voices = len(voicebank_result.speaker_to_voice)
    print(f"      → {n_voices} 个 voice reference  "
          f"(backend={voicebank_result.backend}, success={voicebank_result.success})  "
          f"({t_vb:.2f}s)")
    for name, ref in voicebank_result.speaker_to_voice.items():
        wav_path = Path(ref)
        size_kb = (wav_path.stat().st_size / 1024) if wav_path.exists() else 0
        marker = "OK " if wav_path.exists() else "MISS"
        print(f"      [{marker}] {name:<10} {ref}  ({size_kb:.1f} KB)")

    # 7. generate_director_plan
    t0 = time.time()
    plan = generate_director_plan(
        resolved, characters, llm, story_context=story.story_name
    )
    t_dir = time.time() - t0
    print(f"[7/8] generate_director_plan    → {len(plan)} 条指令  "
          f"(与 segments 数量一致: {len(plan) == len(resolved)})  "
          f"({t_dir:.2f}s)")

    # 8. build_tts_instructions（用真实 voicebank_result）
    t0 = time.time()
    tts_instructions = build_tts_instructions(
        segments=resolved,
        characters=characters,
        director_plan=plan,
        voicebank_result=voicebank_result,
    )
    t_tts = time.time() - t0
    missing_voice = sum(
        1 for i in tts_instructions if not i.metadata.get("has_voice_ref")
    )
    missing_director = sum(
        1 for i in tts_instructions if not i.metadata.get("has_director_instruction")
    )
    print(f"[8/8] build_tts_instructions    → {len(tts_instructions)} 条  "
          f"(缺 director: {missing_director}, 缺 voice_ref: {missing_voice})  "
          f"({t_tts:.2f}s)")
    print()

    # ── 落盘 ─────────────────────────────────────────────────────────────────
    segments_raw_path = json_dir / "segments_raw.json"
    segments_merged_path = json_dir / "segments_after_quote_merge.json"
    resolved_path = json_dir / "resolved_segments.json"
    characters_path = json_dir / "characters.json"
    plan_path = json_dir / "director_plan.json"
    voicebank_result_path = json_dir / "voicebank_result.json"
    tts_path = json_dir / "tts_instructions.json"
    # quote_classifications.json 已经在 step 3 由 quote_classifier 写好

    _write_json(segments_raw_path, _segments_to_dicts(segments_raw))
    _write_json(segments_merged_path, _segments_to_dicts(segments_merged))
    _write_json(resolved_path, _segments_to_dicts(resolved))
    _write_json(characters_path, _characters_to_dicts(characters))
    _write_json(voicebank_result_path, _voicebank_result_to_dict(voicebank_result))
    _write_json(plan_path, _plan_to_dicts(plan))
    _write_json(tts_path, _tts_instructions_to_dicts(tts_instructions))

    # ── 总结 ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("[summary]")
    print(f"  raw segments (build_segments 后)       = {len(segments_raw)}  "
          f"(types={raw_counts})")
    print(f"  after quote merge (classify_and_merge) = {len(segments_merged)}  "
          f"(types={merged_counts}; 合并回 narration: {n_merged_back})")
    print(f"  resolved segments                      = {len(resolved)}  "
          f"(types={resolved_counts})")
    print(f"  dialogue / inner_thought speakers      = {speakers_in_dialogue}")
    print(f"  characters                             = {len(characters)}")
    print(f"  voicebank_result                       = {n_voices} voices  "
          f"(success={voicebank_result.success})")
    print(f"  director_plan                          = {len(plan)}")
    print(f"  tts_instructions                       = {len(tts_instructions)}  "
          f"(缺 director: {missing_director}, 缺 voice_ref: {missing_voice})")
    print()
    print("  角色清单:")
    for c in characters:
        print(
            f"    - {c.name:<10} role={c.role_type:<9} "
            f"gender={str(c.gender):<7} age={str(c.age_style):<12} "
            f"conf={c.confidence:.2f}"
        )
        print(f"      voice_prompt = {c.voice_prompt!r}")
    print()

    # director_plan 前 5 条预览
    print("  director_plan 前 5 条预览:")
    for d in plan[:5]:
        print(
            f"    - {d.segment_id} speaker={d.speaker:<6} "
            f"emotion={d.emotion:<11} intensity={d.emotion_intensity:.2f}  "
            f"pace={d.pace:.2f} tone={d.tone:<8} volume={d.volume:<6} "
            f"pitch={d.pitch:<11} pause={d.pause_hint:.1f}s"
        )
        print(f"      stress_words={d.stress_words}")
        print(f"      delivery={d.delivery_instruction!r}")
    print()

    # tts_instructions 前 3 条预览（验证 Segment + Director + 真实 Voicebank 合并结果）
    print("  tts_instructions 前 3 条预览:")
    for inst in tts_instructions[:3]:
        print(
            f"    - {inst.segment_id} speaker={inst.speaker:<6} "
            f"type={inst.segment_type:<13} "
            f"emotion={inst.emotion:<11} pace={inst.pace:.2f} "
            f"volume={inst.volume:<6} pitch={inst.pitch:<11}"
        )
        print(f"      voice_ref={inst.voice_ref!r}  output={inst.output_filename}")
        print(f"      metadata={inst.metadata}")
    print()

    print("  输出 JSON:")
    for p in (
        segments_raw_path,
        debug_path,
        segments_merged_path,
        resolved_path,
        characters_path,
        voicebank_result_path,
        plan_path,
        tts_path,
    ):
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    - {p}  ({size_kb:.1f} KB)")
        else:
            print(f"    - {p}  (MISSING)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
