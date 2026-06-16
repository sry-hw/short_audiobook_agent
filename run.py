"""Short Audiobook Director Agent - 主入口。

用法：
    python -X utf8 run.py input/sample_story_01.txt
    python -X utf8 run.py input/sample_story_01.txt --skip-tts
    python -X utf8 run.py input/sample_story_01.txt --from-json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_loader import load_text
from story_parser import parse_text
from llm_story_resolver import resolve_quotes
from segment_builder import build_segments
from character_analyzer import analyze_characters
from story_director import direct_story
from tts_instruction_generator import generate_instructions


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def synthesize_only(args, story_name, analysis_dir):
    """基于已有的 JSON 文件，直接合成音频。"""
    t_start = time.perf_counter()

    print(f"\n{'='*50}")
    print(f"快速合成模式：{story_name}")
    print(f"{'='*50}\n")

    # 读取已有的 JSON
    director_plan_path = analysis_dir / f"{story_name}_director_plan.json"
    characters_path = analysis_dir / f"{story_name}_characters.json"

    if not director_plan_path.exists():
        print(f"缺少导演计划文件：{director_plan_path}")
        print("请先运行完整链路生成导演计划")
        return

    print(f"[1/3] 读取导演计划: {director_plan_path}")
    director_plan = load_json(director_plan_path)

    print(f"[2/3] 读取角色信息: {characters_path}")
    characters = load_json(characters_path)

    print(f"[3/3] 生成 TTS 指令...")
    instructions = generate_instructions(director_plan, characters, story_name=story_name)
    save_json(instructions, analysis_dir / f"{story_name}_tts_instructions.json")
    print(f"  共 {instructions['total_instructions']} 条指令")

    print("\n开始合成音频...")
    from tts_engine import synthesize_and_stitch
    final_output = args.final_output or f"output/audio_final/{story_name}_final.wav"
    result = synthesize_and_stitch(
        instructions,
        output_dir=args.output_dir,
        final_output=final_output,
    )
    save_json(result, analysis_dir / f"{story_name}_tts_results.json")

    elapsed = time.perf_counter() - t_start
    synth = result["synthesis"]
    stitch = result["stitch"]

    print(f"\n{'='*50}")
    print(f"完成！")
    print(f"  总耗时：{elapsed:.1f}s")
    print(f"  成功：{synth['success']}/{synth['total']} 片段")
    if synth["failed"] > 0:
        print(f"  失败：{synth['failed']} 片段")
    print(f"  最终音频：{stitch['output_path']}")
    print(f"  音频时长：{stitch['total_duration_seconds']}s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Short Audiobook Director Agent")
    parser.add_argument("input_file", help="输入文本文件路径")
    parser.add_argument("--output-dir", default="output/audio_segments", help="音频片段输出目录")
    parser.add_argument("--final-output", default="", help="最终拼接音频路径")
    parser.add_argument("--skip-tts", action="store_true", help="跳过 TTS 生成，只输出中间分析结果")
    parser.add_argument("--from-json", action="store_true", help="跳过 LLM 链路，直接基于已有的 JSON 文件合成音频")
    args = parser.parse_args()

    story_name = Path(args.input_file).stem
    analysis_dir = Path("output/analysis")

    if args.from_json:
        synthesize_only(args, story_name, analysis_dir)
        return

    t_start = time.perf_counter()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"正在处理：{args.input_file}")
    print(f"{'='*50}\n")

    print("[1/7] 读取文本...")
    text = load_text(args.input_file)
    print(f"  读取完成，共 {len(text)} 字")

    print("\n[2/7] 解析文本结构...")
    parsed = parse_text(text)
    print(f"  共 {parsed['total_paragraphs']} 个段落")

    print("\n[3/7] LLM 语义判断...")
    resolved = resolve_quotes(parsed)
    print(f"  共判断 {resolved['total_resolved']} 个引号")

    print("\n[4/7] 构建 segments...")
    segments = build_segments(parsed, resolved)
    save_json(segments, analysis_dir / f"{story_name}_segments.json")
    print(f"  共 {segments['total_segments']} 个 segment")

    print("\n[5/7] 分析角色声音特征...")
    characters = analyze_characters(segments, text)
    save_json(characters, analysis_dir / f"{story_name}_characters.json")
    char_names = [c["speaker"] for c in characters.get("characters", [])]
    print(f"  识别角色：{', '.join(char_names) if char_names else '无'} + narrator")

    print("\n[6/7] 生成导演计划...")
    director_plan = direct_story(segments, characters, text)
    save_json(director_plan, analysis_dir / f"{story_name}_director_plan.json")
    print(f"  共 {len(director_plan.get('segment_directions', []))} 个片段导演指导")

    print("\n[7/7] 生成 TTS 指令...")
    instructions = generate_instructions(director_plan, characters, story_name=story_name)
    save_json(instructions, analysis_dir / f"{story_name}_tts_instructions.json")
    print(f"  共 {instructions['total_instructions']} 条指令")

    if args.skip_tts:
        elapsed = time.perf_counter() - t_start
        print(f"\n{'='*50}")
        print(f"完成（跳过 TTS），耗时 {elapsed:.1f}s")
        print(f"中间结果已保存到 {analysis_dir}/")
        return

    print("\n开始合成音频...")
    from tts_engine import synthesize_and_stitch
    final_output = args.final_output or f"output/audio_final/{story_name}_final.wav"
    result = synthesize_and_stitch(
        instructions,
        output_dir=args.output_dir,
        final_output=final_output,
    )
    save_json(result, analysis_dir / f"{story_name}_tts_results.json")

    elapsed = time.perf_counter() - t_start
    synth = result["synthesis"]
    stitch = result["stitch"]

    print(f"\n{'='*50}")
    print(f"完成！")
    print(f"  总耗时：{elapsed:.1f}s")
    print(f"  成功：{synth['success']}/{synth['total']} 片段")
    if synth["failed"] > 0:
        print(f"  失败：{synth['failed']} 片段")
    print(f"  最终音频：{stitch['output_path']}")
    print(f"  音频时长：{stitch['total_duration_seconds']}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()