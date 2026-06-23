"""src_next/tts/test_tts_from_artifacts.py

基于已有 analysis 产物跑 TTS adapter 的 smoke 测试脚本。

**不重新跑 analysis，不重新生成 voicebank**，直接读取已经落盘的：
    <artifact_dir>/json/tts_instructions.json
    <artifact_dir>/json/voicebank_result.json

调用指定 backend 的 TTS adapter 合成分段 wav，落到：
    <artifact_dir>/audio_segments/

并把每段结果汇总到：
    <artifact_dir>/json/audio_segment_results.json

用法（详见 ``--help``）：

    # mock：不调用真实模型，验证数据流
    python -m src_next.tts.test_tts_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨 \\
        --backend mock \\
        --dry-run true

    # IndexTTS dry-run：只写 invocation + style snapshot 到 log，不调用模型
    python -m src_next.tts.test_tts_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨 \\
        --backend indextts \\
        --dry-run true \\
        --limit 2

    # IndexTTS 真实合成：少量条目先验证
    python -m src_next.tts.test_tts_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨 \\
        --backend indextts \\
        --dry-run false \\
        --limit 2

    # 全量
    python -m src_next.tts.test_tts_from_artifacts \\
        --artifact-dir output-src-next-analysis-test/桂花雨 \\
        --backend indextts \\
        --dry-run false \\
        --limit 0
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


# Windows GBK 终端兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


from src_next.core.data_models import (  # noqa: E402
    AudioSegmentResult,
    TTSInstruction,
    VoicebankResult,
)
from src_next.tts.registry import create_tts_adapter  # noqa: E402


_DEFAULT_PROFILE = "src_next/profiles/blue_indextts.yaml"


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于已有 analysis 产物跑 TTS adapter（不重跑 analysis、不重生 voicebank）"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="故事级输出根目录（如 output-src-next-analysis-test/桂花雨）",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "indextts", "cosyvoice", "fishpro", "qwen_tts"],
        help="TTS 后端（默认 mock；当前只有 mock / indextts 真实可用）",
    )
    parser.add_argument(
        "--dry-run",
        default="true",
        choices=["true", "false"],
        help="true=不真实调用模型；false=真实合成",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 条（0=全量）",
    )
    parser.add_argument(
        "--profile",
        default=_DEFAULT_PROFILE,
        help=f"indextts backend 的 profile 路径（默认 {_DEFAULT_PROFILE}）",
    )
    return parser.parse_args()


# ── 反序列化 ────────────────────────────────────────────────────────────────

def _load_tts_instructions(path: Path) -> list[TTSInstruction]:
    """从 tts_instructions.json 读回 TTSInstruction 列表。

    dataclasses.asdict 出来的 dict 可以直接 ** 解构回 dataclass，
    只要字段名一一对应（目前是一一对应的）。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} 顶层不是 list")
    instructions: list[TTSInstruction] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # 只挑已知字段，多余字段忽略（前向兼容）
        fields = {f.name for f in dataclasses.fields(TTSInstruction)}
        kwargs = {k: v for k, v in item.items() if k in fields}
        instructions.append(TTSInstruction(**kwargs))
    return instructions


def _load_voicebank_result(path: Path) -> VoicebankResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(VoicebankResult)}
    kwargs = {k: v for k, v in raw.items() if k in fields}
    return VoicebankResult(**kwargs)


# ── adapter 构造 ────────────────────────────────────────────────────────────

def _load_tts_profile(profile_path: Path) -> tuple[str, dict[str, Any]]:
    """从 yaml profile 加载 tts 配置段。

    Returns:
        (backend, config_dict)
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    tts = profile.get("tts") or {}
    backend = tts.pop("backend")
    return backend, tts


def _build_adapter(backend: str, profile_path: Path) -> Any:
    """根据 backend 创建 adapter；indextts 走 profile，其它走默认。"""
    if backend == "indextts":
        if not profile_path.exists():
            print(f"[ERROR] indextts profile 不存在: {profile_path}", file=sys.stderr)
            sys.exit(2)
        prof_backend, config = _load_tts_profile(profile_path)
        if prof_backend != backend:
            print(
                f"[WARN] profile backend={prof_backend!r} 与 --backend={backend!r} 不一致，"
                "按 CLI backend 走，但用 profile 的 config",
                file=sys.stderr,
            )
        return create_tts_adapter(backend, **config)
    # mock / 占位 backend 不需要 config
    return create_tts_adapter(backend)


# ── 写盘 ────────────────────────────────────────────────────────────────────

def _results_to_dicts(results: list[AudioSegmentResult]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(r) for r in results]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    json_dir = artifact_dir / "json"
    instructions_path = json_dir / "tts_instructions.json"
    voicebank_path = json_dir / "voicebank_result.json"

    if not instructions_path.exists():
        print(f"[ERROR] 缺 {instructions_path}", file=sys.stderr)
        return 2
    if not voicebank_path.exists():
        print(f"[ERROR] 缺 {voicebank_path}（先用 analysis smoke test 生成）", file=sys.stderr)
        return 2

    profile_path = Path(args.profile).expanduser().resolve()
    dry_run = args.dry_run == "true"
    limit = args.limit

    print("=" * 60)
    print("[tts from artifacts] 配置")
    print(f"  artifact_dir   = {artifact_dir}")
    print(f"  backend        = {args.backend}")
    print(f"  dry_run        = {dry_run}")
    print(f"  limit          = {limit if limit > 0 else 'all'}")
    if args.backend == "indextts":
        print(f"  profile        = {profile_path}")
    print("=" * 60)
    print()

    # 1. 读 artifacts
    instructions = _load_tts_instructions(instructions_path)
    voicebank_result = _load_voicebank_result(voicebank_path)
    print(f"[1/3] 加载 artifacts")
    print(f"      tts_instructions = {len(instructions)} 条")
    print(f"      voicebank_result = {len(voicebank_result.speaker_to_voice)} voices "
          f"(backend={voicebank_result.backend}, success={voicebank_result.success})")
    for name, ref in voicebank_result.speaker_to_voice.items():
        wav_path = Path(ref)
        size_kb = (wav_path.stat().st_size / 1024) if wav_path.exists() else 0
        marker = "OK " if wav_path.exists() else "MISS"
        print(f"      [{marker}] {name:<10} {ref}  ({size_kb:.1f} KB)")

    # 2. 构造 adapter
    print(f"[2/3] 构造 adapter ({args.backend}) ...")
    try:
        adapter = _build_adapter(args.backend, profile_path)
    except Exception as err:  # noqa: BLE001
        print(f"[ERROR] adapter 构造失败: {type(err).__name__}: {err}", file=sys.stderr)
        return 3

    # 3. synthesize
    print(f"[3/3] synthesize ... (dry_run={dry_run}, limit={limit})")
    t0 = time.time()
    try:
        results = adapter.synthesize(
            instructions,
            voicebank_result,
            str(artifact_dir),
            dry_run=dry_run,
            limit=limit,
        )
    except Exception as err:  # noqa: BLE001
        print(f"[ERROR] synthesize 失败: {type(err).__name__}: {err}", file=sys.stderr)
        return 4
    elapsed = time.time() - t0

    success_n = sum(1 for r in results if r.success)
    fail_n = len(results) - success_n
    print(f"      → {len(results)} 条结果  (success={success_n}, fail={fail_n})  "
          f"({elapsed:.2f}s)")
    print()

    # ── 落盘 ─────────────────────────────────────────────────────────
    audio_dir = artifact_dir / "audio_segments"
    results_path = json_dir / "audio_segment_results.json"
    _write_json(results_path, _results_to_dicts(results))

    # ── 总结 ────────────────────────────────────────────────────────
    print("=" * 60)
    print("[summary]")
    print(f"  instructions     = {len(instructions)}")
    print(f"  results          = {len(results)}  (success={success_n}, fail={fail_n})")
    print(f"  audio_segments/  = {audio_dir}")
    print(f"  results json     = {results_path}")
    print()

    # 失败列表
    failures = [r for r in results if not r.success]
    if failures:
        print(f"  失败 {len(failures)} 条:")
        for r in failures[:10]:
            print(f"    - {r.segment_id} speaker={r.speaker:<6} error={r.error}")
        if len(failures) > 10:
            print(f"    ... 共 {len(failures)} 条（只显示前 10 条）")
        print()

    # 成功的 wav 大小预览（前 5 条）
    successes = [r for r in results if r.success and r.audio_path]
    if successes:
        print(f"  成功 wav 预览（前 5 条）:")
        for r in successes[:5]:
            p = Path(r.audio_path)
            if p.exists() and not r.audio_path.startswith("mock://"):
                size_kb = p.stat().st_size / 1024
                print(f"    - {r.segment_id} {r.audio_path}  ({size_kb:.1f} KB)")
            else:
                print(f"    - {r.segment_id} {r.audio_path}")
        print()

    print(f"  完整日志/错误: {audio_dir}/errors.log （若存在）")
    print(f"  adapter config snapshot: {audio_dir}/adapter_config.json")
    print("=" * 60)

    # dry_run 模式下所有 instruction 都 "失败" 是预期行为，不视为错误
    if dry_run:
        return 0
    return 0 if fail_n == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
