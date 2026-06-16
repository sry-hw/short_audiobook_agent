"""IndexTTS + Qwen3 链路入口。

用法：
    python run.py input/sample_story_01.txt --full
    python run.py input/sample_story_01.txt --stage voicebank
    python run.py input/sample_story_01.txt --stage director
    python run.py input/sample_story_01.txt --stage instructions
    python run.py input/sample_story_01.txt --skip-tts
    python run.py input/sample_story_01.txt --from-json
    python run.py input/sample_story_01.txt --full --force
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

# 添加 src 目录到路径，以复用现有模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_loader import load_text
from story_parser import parse_text
from llm_story_resolver import resolve_quotes
from segment_builder import build_segments
from character_analyzer import analyze_characters

from indextts.story_director import direct_story
from indextts.voice_bank_generator import generate_voice_bank
from indextts.tts_instruction_generator import generate_instructions
from indextts.tts_engine import synthesize_and_stitch


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# 各阶段入口
# ----------------------------------------------------------------------


def stage_characters(args, story_name, out_dir):
    """阶段1：角色分析"""
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    print("    [1/4] 加载文本")
    text = load_text(args.input_file)
    print("    [2/4] 解析段落与引号结构")
    parsed = parse_text(text)
    print("    [3/4] LLM 语义判断（引号类型+说话人）")
    resolved = resolve_quotes(parsed)
    print("    [4/4] LLM 角色声音分析")
    segments = build_segments(parsed, resolved)
    characters = analyze_characters(segments, text)
    save_json(characters, json_dir / f"{story_name}_characters.json")
    print(f"    ✓ narrator + {[c['speaker'] for c in characters.get('characters', [])]}，共 {len(characters.get('characters', []))} 个角色")
    return characters


def stage_voicebank(args, story_name, out_dir):
    """阶段2：Qwen3 生成音色参考"""
    json_dir = out_dir / "json"
    voicebank_dir = out_dir / "voicebank"
    characters = load_json(json_dir / f"{story_name}_characters.json")
    voice_bank = generate_voice_bank(characters, str(voicebank_dir), force=args.force)
    return voice_bank


def stage_director(args, story_name, out_dir):
    """阶段3：生成导演计划"""
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    print("    [1/5] 加载文本")
    text = load_text(args.input_file)
    print("    [2/5] 解析段落与引号结构")
    parsed = parse_text(text)
    print("    [3/5] LLM 语义判断（引号类型+说话人）")
    resolved = resolve_quotes(parsed)
    print("    [4/5] 构建 segments")
    segments = build_segments(parsed, resolved)
    print("    [5/5] LLM 导演计划（IndexTTS 参数）")
    characters = load_json(json_dir / f"{story_name}_characters.json")
    director_plan = direct_story(segments, characters, text)
    save_json(director_plan, json_dir / f"{story_name}_director_plan.json")

    style = director_plan.get("overall_style", {})
    print(f"    ✓ genre={style.get('genre', '')}, tone={style.get('tone', '')}, pace={style.get('pace', '')}")
    print(f"    ✓ {len(director_plan.get('segment_directions', []))} 个 segment 导演指导")
    return director_plan


def stage_instructions(args, story_name, out_dir):
    """阶段4：生成 TTS 指令"""
    json_dir = out_dir / "json"
    voicebank_dir = out_dir / "voicebank"

    characters = load_json(json_dir / f"{story_name}_characters.json")
    director_plan = load_json(json_dir / f"{story_name}_director_plan.json")

    voice_bank = {}
    for c in characters.get("characters", []):
        voice_bank[c["speaker"]] = str(voicebank_dir / f"{c['speaker']}.wav")
    voice_bank["narrator"] = str(voicebank_dir / "narrator.wav")

    instructions = generate_instructions(director_plan, characters, voice_bank, story_name=story_name)
    save_json(instructions, json_dir / f"{story_name}_tts_instructions.json")
    print(f"    ✓ {instructions['total_instructions']} 条指令已生成")
    return instructions


def stage_audio(args, story_name, out_dir):
    """阶段5：合成音频"""
    json_dir = out_dir / "json"
    voicebank_dir = out_dir / "voicebank"

    instructions = load_json(json_dir / f"{story_name}_tts_instructions.json")
    audio_dir = out_dir / "audio_segments"
    final_dir = out_dir / "audio_final"
    final_path = final_dir / f"{story_name}_final.wav"

    result = synthesize_and_stitch(
        instructions,
        voice_bank_dir=str(voicebank_dir),
        output_dir=str(audio_dir),
        final_output=str(final_path),
    )
    save_json(result, json_dir / f"{story_name}_tts_results.json")

    synth = result["synthesis"]
    stitch = result["stitch"]
    print(f"    ✓ 成功 {synth['success']}/{synth['total']} 片段", end="")
    if synth["failed"] > 0:
        print(f"，失败 {synth['failed']} 片段")
    else:
        print()
    print(f"    ✓ 最终音频：{stitch['output_path']} ({stitch['total_duration_seconds']}s)")
    return result


# ----------------------------------------------------------------------
# 主入口 & 模式判断
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="IndexTTS + Qwen3 有声书 TTS 链路")
    parser.add_argument("input_file", help="输入文本文件路径")
    parser.add_argument("--force", action="store_true", help="强制重生成，跳过已有文件")
    parser.add_argument("--skip-tts", action="store_true", help="生成 JSON 但不合成音频")
    parser.add_argument("--from-json", action="store_true", help="基于已有 JSON 直接合成音频")
    parser.add_argument("--stage",
        choices=["characters", "voicebank", "director", "instructions", "audio"],
        default=None,
        help="从指定阶段继续")
    parser.add_argument("--full", action="store_true", help="完整链路")
    args = parser.parse_args()

    story_name = Path(args.input_file).stem
    out_dir = Path("output-indextts") / story_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    if args.from_json:
        _run_from_json(args, story_name, out_dir)
        return

    stage = args.stage
    if stage is None and not args.full:
        parser.print_help()
        return

    stages_order = ["characters", "voicebank", "director", "instructions", "audio"]

    if args.full:
        run_stages = stages_order
    else:
        run_stages = stages_order[stages_order.index(stage):]

    _print_header(story_name, run_stages)

    for i, s in enumerate(run_stages):
        print(f"\n[{i+1}/{len(run_stages)}] {STAGE_LABELS[s]}...")
        if s == "characters":
            stage_characters(args, story_name, out_dir)
        elif s == "voicebank":
            stage_voicebank(args, story_name, out_dir)
        elif s == "director":
            stage_director(args, story_name, out_dir)
        elif s == "instructions":
            stage_instructions(args, story_name, out_dir)
        elif s == "audio":
            if args.skip_tts:
                print("    跳过 TTS 合成")
            else:
                stage_audio(args, story_name, out_dir)

    elapsed = time.perf_counter() - t_start
    _print_footer(story_name, out_dir, elapsed)


def _run_from_json(args, story_name, out_dir):
    """基于已有 JSON 直接合成音频（不重跑 LLM/Qwen3）。"""
    t_start = time.perf_counter()
    _print_header(story_name, ["audio"])

    print(f"\n[1/1] 合成音频（从已有 JSON）...")
    stage_audio(args, story_name, out_dir)

    elapsed = time.perf_counter() - t_start
    _print_footer(story_name, out_dir, elapsed)


STAGE_LABELS = {
    "characters": "角色分析",
    "voicebank": "生成音色参考（Qwen3-VoiceDesign）",
    "director": "生成导演计划（LLM，IndexTTS 参数）",
    "instructions": "生成 TTS 指令（IndexTTS）",
    "audio": "合成音频（IndexTTS）",
}


def _print_header(story_name, stages):
    stages_str = " → ".join(STAGE_LABELS[s] for s in stages)
    print(f"\n{'='*60}")
    print(f"IndexTTS + Qwen3 TTS 链路")
    print(f"  故事：{story_name}")
    print(f"  阶段：{stages_str}")
    print(f"  输出：output-indextts/{story_name}/")
    print(f"    ├── json/          (中间 JSON)")
    print(f"    ├── voicebank/     (音色参考 WAV)")
    print(f"    ├── audio_segments/(合成片段)")
    print(f"    └── audio_final/   (最终音频)")
    print(f"{'='*60}")


def _print_footer(story_name, out_dir, elapsed):
    print(f"\n{'='*60}")
    print(f"✓ 完成！总耗时：{elapsed:.1f}s")
    print(f"  中间 JSON：output-indextts/{story_name}/json/")
    print(f"  音色参考：output-indextts/{story_name}/voicebank/")
    print(f"  最终音频：output-indextts/{story_name}/audio_final/{story_name}_final.wav")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()