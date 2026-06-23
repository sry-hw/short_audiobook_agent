"""src_next/core/test_audio_merger_from_artifacts.py

基于已有 TTS 产物跑 audio_merger 的 smoke 脚本。

不重跑 analysis、不重跑 TTS，直接读：
    <artifact_dir>/json/audio_segment_results.json
    <artifact_dir>/json/tts_instructions.json     ← 取 pause_hint 做段间静音

调 ``core.audio_merger.merge_audio_segments`` 把所有 success=True 且
文件存在的 wav 按 segment 顺序拼成单个 wav，段间按 ``pause_hint`` 插入静音：

    <artifact_dir>/audio_final/<story_name>.wav

并把 ``AudioResult`` 落到：
    <artifact_dir>/json/audio_result.json

用法：
    python -m src_next.core.test_audio_merger_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨

    # 关闭段间静音（纯拼接，调试用）
    python -m src_next.core.test_audio_merger_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨 \\
        --no-pause
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from pathlib import Path


# Windows GBK 终端兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


from src_next.core.audio_merger import merge_audio_segments  # noqa: E402
from src_next.core.data_models import AudioSegmentResult, TTSInstruction  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 audio_segment_results.json 拼接最终 wav（支持 pause_hint 段间静音）"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="故事级输出根目录（如 output-src-next-analysis-test/桂花雨）",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="最终 wav 文件名（默认 <story_name>.wav）",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="关闭段间静音（默认从 tts_instructions.json 读 pause_hint）",
    )
    return parser.parse_args()


def _load_audio_results(path: Path) -> list[AudioSegmentResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} 顶层不是 list")
    fields = {f.name for f in dataclasses.fields(AudioSegmentResult)}
    segments: list[AudioSegmentResult] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kwargs = {k: v for k, v in item.items() if k in fields}
        segments.append(AudioSegmentResult(**kwargs))
    return segments


def _load_pause_hints(path: Path) -> dict[str, float]:
    """从 tts_instructions.json 读 segment_id → pause_hint 映射。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {}
    out: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        seg_id = item.get("segment_id")
        if not seg_id:
            continue
        try:
            out[seg_id] = float(item.get("pause_hint", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    results_path = artifact_dir / "json" / "audio_segment_results.json"
    instructions_path = artifact_dir / "json" / "tts_instructions.json"
    if not results_path.exists():
        print(f"[ERROR] 缺 {results_path}", file=sys.stderr)
        return 2

    story_name = artifact_dir.name
    output_name = args.output_name or f"{story_name}.wav"

    final_dir = artifact_dir / "audio_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / output_name

    # 1. 读 segments
    segments = _load_audio_results(results_path)
    total = len(segments)
    success_n = sum(1 for s in segments if s.success)
    real_wav_n = sum(
        1 for s in segments
        if s.success and s.audio_path and not s.audio_path.startswith("mock://")
        and Path(s.audio_path).exists()
    )

    # 2. 读 pause_hint（除非 --no-pause）
    pause_map: dict[str, float] | None = None
    pause_total = 0.0
    if not args.no_pause:
        if instructions_path.exists():
            pause_map = _load_pause_hints(instructions_path)
            # 只对实际能拼进 wav 的段统计静音总量
            usable_ids = {
                s.segment_id for s in segments
                if s.success and s.audio_path
                and not s.audio_path.startswith("mock://")
                and Path(s.audio_path).exists()
            }
            pause_total = sum(v for k, v in pause_map.items() if k in usable_ids)
        else:
            print(f"[WARN] 没找到 {instructions_path}，段间不插静音", file=sys.stderr)

    print("=" * 60)
    print("[audio_merger from artifacts] 配置")
    print(f"  artifact_dir   = {artifact_dir}")
    print(f"  story_name     = {story_name}")
    print(f"  final_path     = {final_path}")
    print(f"  segments       = {total}  (success={success_n}, real_wav={real_wav_n})")
    if pause_map is not None:
        print(f"  pause_mode     = pause_hint (from tts_instructions.json)")
        print(f"  pause_total    = {pause_total:.2f}s across usable segments")
    else:
        print(f"  pause_mode     = off (--no-pause)")
    print("=" * 60)
    print()

    # 3. 调 merger
    audio_result = merge_audio_segments(segments, str(final_path), pause_seconds_after=pause_map)

    # 4. 落 audio_result.json
    audio_result_path = artifact_dir / "json" / "audio_result.json"
    audio_result_path.write_text(
        json.dumps(dataclasses.asdict(audio_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 5. 总结
    print("[summary]")
    print(f"  success        = {audio_result.success}")
    print(f"  final_audio    = {audio_result.final_audio}")
    print(f"  duration       = {audio_result.duration_seconds:.2f}s")
    if audio_result.final_audio and Path(audio_result.final_audio).exists():
        size_mb = Path(audio_result.final_audio).stat().st_size / (1024 * 1024)
        print(f"  size           = {size_mb:.2f} MB")
    print(f"  audio_result   = {audio_result_path}")
    print("=" * 60)

    return 0 if audio_result.success else 5


if __name__ == "__main__":
    sys.exit(main())
