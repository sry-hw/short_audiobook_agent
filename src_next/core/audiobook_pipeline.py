"""src_next/core/audiobook_pipeline.py

有声书 Agent 主流程编排层。两个入口：

* ``run_mock_core_pipeline(input_path, output_root)``
  离线 mock 链路，characters / director_plan / voicebank / audio_segments 全是
  假数据。用于验证 core 层数据流和目录结构，不依赖 LLM / GPU / 任何外部服务。

* ``run_pipeline(input_path, profile, output_root=None, story_name=None)``
  真实端到端链路，由 profile yaml 驱动：
      txt → segments → merged → resolved → characters → voicebank →
            director_plan → tts_instructions → audio_segments → final wav
  调用 analysis / voicebank / tts / llm 四个适配层的真实后端。

CLI（``python -m src_next.core.audiobook_pipeline``）：
    --input <path>             必填
    --profile <path>           必填（除非 --mock）
    [--output-root <path>]     默认 profile['output']['root']
    [--story-name <name>]      默认 input 文件 stem
    [--mock]                   走 mock pipeline，忽略 --profile
    [--reuse-existing]         覆盖 profile.pipeline.reuse_existing=true

Console 日志约定：
    [Pipeline] input=...
    [Pipeline] profile=...
    [Pipeline] output=...

    [1/10] build_segments ... done in 0.02s, segments=21
    [2/10] create_llm_client ... done in 0.00s, backend=qwen_http, model=qwen3.6-plus
    [3/10] quote_classifier ... done in 45.32s, segments_after_merge=17
    ...
    [9/10] tts_synthesis ... done in 38.50s, success=17/17
    [10/10] audio_merger ... done in 0.30s, final=audio_final/<story>.wav

    [Summary]
    success=true
    total_time=223.10s
    analysis_time=148.00s
    voicebank_time=36.10s
    tts_time=38.50s
    merge_time=0.30s
    final_audio_duration=167.20s
    rtf=1.33
    output_dir=...
    final_audio=...

失败 / 缓存 / 复用 三态分别打印：
    failed in X.XXs — <error>
    reused, loaded from <file>, X.XXs
    done in X.XXs, success=N/M, cached
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import asdict, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

# logging_utils 在 import 时会把 stdout 切到 UTF-8（Windows GBK 兼容），
# 早于本模块任何 print 调用，所以不必再做 wrapper。
from .audio_merger import merge_audio_segments
from .data_models import (
    AudioResult,
    AudioSegmentResult,
    CharacterProfile,
    DirectorInstruction,
    PipelineResult,
    Segment,
    StoryInput,
    TTSInstruction,
    VoicebankResult,
)
from .logging_utils import log_item, log_stage_done, log_stage_start
from .segment_builder import build_segments
from .tts_instruction_builder import build_tts_instructions

# 真实 pipeline 用的 analysis + adapter 工厂
from ..analysis.quote_classifier import classify_and_merge_quotes
from ..analysis.story_resolver import resolve_speakers
from ..analysis.character_analyzer import analyze_characters
from ..analysis.story_director import generate_director_plan
from ..llm.registry import create_llm_client
from ..tts.registry import create_tts_adapter
from ..voicebank.registry import create_voicebank_adapter


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具（mock 和真实 pipeline 共用）
# ─────────────────────────────────────────────────────────────────────────────

def _story_name_from_path(input_path: str) -> str:
    """从输入路径提取 story_name（不带扩展名的文件名）。"""
    return Path(input_path).stem


def _serialize(obj: Any) -> Any:
    """递归把 dataclass / list / dict 转成 JSON 可序列化结构。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _save_json(obj: Any, path: Path) -> None:
    """保存为 UTF-8 JSON（缩进 2，保留中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_serialize(obj), f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Profile loader + backend block 拆分
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_BLOCKS: tuple[str, ...] = ("llm", "voicebank", "tts", "output", "pipeline")


def _load_pipeline_profile(path: str | Path) -> dict[str, Any]:
    """yaml.safe_load + 5 块校验。返回原始 dict（含 backend 字段）。

    Raises:
        FileNotFoundError: path 不存在。
        ValueError: 缺块 / 缺 backend / output.root 缺失 / pipeline 三个 flag 缺失。
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"profile 不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profile 顶层不是 dict: {p}")
    for block in _REQUIRED_BLOCKS:
        if block not in data:
            raise ValueError(f"profile 缺块: {block!r}（file={p}）")
        if not isinstance(data[block], dict):
            raise ValueError(f"profile 块 {block!r} 不是 dict（file={p}）")
    for block in ("llm", "voicebank", "tts"):
        if not data[block].get("backend"):
            raise ValueError(f"profile 块 {block!r} 缺 backend 字段（file={p}）")
    if not data["output"].get("root"):
        raise ValueError(f"profile 块 'output' 缺 root 字段（file={p}）")
    pipeline_block = data["pipeline"]
    for flag in ("save_intermediate_json", "reuse_existing", "stop_on_tts_error"):
        if flag not in pipeline_block:
            raise ValueError(f"profile.pipeline 缺 {flag!r}（file={p}）")
    return data


def _split_backend_block(profile: dict[str, Any], block: str) -> tuple[str, dict[str, Any]]:
    """从 profile[block] pop 出 backend，返回 (backend, 剩余 config 字典)。

    剩余字典正是 ``create_*_client/adapter(backend, **config)`` 期望的形态。
    不会修改原 profile（浅拷贝）。
    """
    block_dict = dict(profile[block])
    backend = block_dict.pop("backend")
    return backend, block_dict


# ─────────────────────────────────────────────────────────────────────────────
# Reuse loaders（从 JSON 反序列化回 dataclass）
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_json(path: Path, loader: Callable[[Path], Any]) -> Any:
    """path 存在 → loader(path)；否则 None。"""
    if path.exists():
        return loader(path)
    return None


def _dataclass_from_list(path: Path, cls: type) -> list[Any]:
    """从 JSON list 还原 dataclass 列表，过滤未知字段。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    allowed = {f.name for f in dataclass_fields(cls)}
    out: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kwargs = {k: v for k, v in item.items() if k in allowed}
        out.append(cls(**kwargs))
    return out


def _load_segments(path: Path) -> list[Segment]:
    return _dataclass_from_list(path, Segment)


def _load_characters(path: Path) -> list[CharacterProfile]:
    return _dataclass_from_list(path, CharacterProfile)


def _load_director_plan(path: Path) -> list[DirectorInstruction]:
    return _dataclass_from_list(path, DirectorInstruction)


def _load_voicebank_result(path: Path) -> VoicebankResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {f.name for f in dataclass_fields(VoicebankResult)}
    kwargs = {k: v for k, v in raw.items() if k in allowed} if isinstance(raw, dict) else {}
    return VoicebankResult(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Console logger（实时进度）
# ─────────────────────────────────────────────────────────────────────────────

def _log_pipeline_header(input_path: str, profile_path: str | None, output_dir: Path) -> None:
    print(f"[Pipeline] input={input_path}", flush=True)
    if profile_path:
        print(f"[Pipeline] profile={profile_path}", flush=True)
    print(f"[Pipeline] output={output_dir}", flush=True)
    print("", flush=True)


def _log_stage_start(step: str, name: str) -> None:
    print(f"[{step}] {name} ...", flush=True)


def _log_stage_done(step: str, name: str, elapsed: float, extra: str = "") -> None:
    suffix = f", {extra}" if extra else ""
    print(f"[{step}] {name} ... done in {elapsed:.2f}s{suffix}", flush=True)


def _log_stage_reused(step: str, name: str, src: str, elapsed: float) -> None:
    print(f"[{step}] {name} ... reused, loaded from {src}, {elapsed:.2f}s", flush=True)


def _log_stage_failed(step: str, name: str, elapsed: float, err: str) -> None:
    print(f"[{step}] {name} ... failed in {elapsed:.2f}s — {err}", flush=True)


def _log_summary(
    *,
    success: bool,
    total_time: float,
    analysis_time: float,
    voicebank_time: float,
    tts_time: float,
    merge_time: float,
    final_audio_duration: float | None,
    rtf: float | None,
    output_dir: str,
    final_audio: str | None,
    error: str = "",
) -> None:
    print("", flush=True)
    print("[Summary]", flush=True)
    print(f"success={'true' if success else 'false'}", flush=True)
    print(f"total_time={total_time:.2f}s", flush=True)
    print(f"analysis_time={analysis_time:.2f}s", flush=True)
    print(f"voicebank_time={voicebank_time:.2f}s", flush=True)
    print(f"tts_time={tts_time:.2f}s", flush=True)
    print(f"merge_time={merge_time:.2f}s", flush=True)
    if final_audio_duration is not None:
        print(f"final_audio_duration={final_audio_duration:.2f}s", flush=True)
    else:
        print("final_audio_duration=null", flush=True)
    if rtf is not None:
        print(f"rtf={rtf:.2f}", flush=True)
    else:
        print("rtf=null", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"final_audio={final_audio or ''}", flush=True)
    if error:
        print(f"error={error}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage record + pipeline_summary 构造
# ─────────────────────────────────────────────────────────────────────────────

def _append_stage_record(
    stages_list: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    elapsed: float,
    mode: str,
    output: str = "",
    error: str | None = None,
) -> None:
    rec: dict[str, Any] = {
        "stage": name,
        "status": status,
        "elapsed_sec": round(elapsed, 3),
        "mode": mode,
        "output": output,
    }
    if error:
        rec["error"] = error
    stages_list.append(rec)


def _read_wav_duration_safe(path: Path) -> float | None:
    """读 wav 时长；不存在或格式错误 → None。"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with wave.open(str(path), "rb") as wf:
            nframes = wf.getnframes()
            framerate = wf.getframerate()
            if not framerate:
                return None
            return nframes / framerate
    except Exception:
        return None


def _build_pipeline_summary(
    *,
    stages: list[dict[str, Any]],
    total_time: float,
    output_dir: str,
    final_audio_path: str | None,
) -> dict[str, Any]:
    """按 stage 名聚合各阶段耗时；算 RTF。"""
    by_name = {s["stage"]: s.get("elapsed_sec", 0.0) for s in stages}
    analysis_time = sum(by_name.get(n, 0.0) for n in (
        "quote_classifier", "story_resolver", "character_analyzer", "story_director",
    ))
    voicebank_time = by_name.get("voicebank", 0.0)
    tts_time = by_name.get("tts_synthesis", 0.0)
    merge_time = by_name.get("audio_merger", 0.0)

    final_duration: float | None = None
    rtf: float | None = None
    if final_audio_path:
        final_duration = _read_wav_duration_safe(Path(final_audio_path))
        if final_duration and final_duration > 0:
            rtf = total_time / final_duration

    return {
        "total_time_sec": round(total_time, 3),
        "analysis_time_sec": round(analysis_time, 3),
        "voicebank_time_sec": round(voicebank_time, 3),
        "tts_time_sec": round(tts_time, 3),
        "merge_time_sec": round(merge_time, 3),
        "final_audio_duration_sec": round(final_duration, 3) if final_duration is not None else None,
        "rtf": round(rtf, 3) if rtf is not None else None,
        "output_dir": output_dir,
        "final_audio_path": final_audio_path or "",
        "stages": stages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 真实端到端 pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    profile: dict[str, Any] | str | Path,
    *,
    output_root: str | None = None,
    story_name: str | None = None,
    reuse_existing_override: bool | None = None,
) -> PipelineResult:
    """运行真实端到端 pipeline。

    Args:
        input_path: 输入 txt 路径。
        profile: dict 或 yaml 文件路径。str/Path 时调 _load_pipeline_profile。
        output_root: 覆盖 profile['output']['root']。None → 用 profile 值。
        story_name: 覆盖 input 文件 stem。None → Path(input_path).stem。
        reuse_existing_override: 覆盖 profile.pipeline.reuse_existing。
            None → 用 profile 值。True/False → 强制覆盖（CLI --reuse-existing 用）。

    Returns:
        PipelineResult；失败时 success=False，error 写明原因，pipeline_summary
        含失败前的所有 stage 统计。
    """
    t_total_start = time.time()

    # ── 解析 profile + 路径 ──────────────────────────────────────────
    profile_path_str: str | None = None
    if isinstance(profile, dict):
        profile_dict = profile
    else:
        profile_path_str = str(profile)
        profile_dict = _load_pipeline_profile(profile)

    story_name = story_name or _story_name_from_path(input_path)
    output_root_resolved = output_root or profile_dict["output"]["root"]
    output_dir = Path(output_root_resolved).expanduser().resolve() / story_name
    json_dir = output_dir / "json"
    audio_final_dir = output_dir / "audio_final"
    json_dir.mkdir(parents=True, exist_ok=True)
    audio_final_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = profile_dict["pipeline"]
    save_json = bool(pipeline_cfg.get("save_intermediate_json", True))
    reuse = bool(pipeline_cfg.get("reuse_existing", False))
    if reuse_existing_override is not None:
        reuse = reuse_existing_override
    stop_on_tts_error = bool(pipeline_cfg.get("stop_on_tts_error", False))

    _log_pipeline_header(input_path, profile_path_str, output_dir)

    # 状态容器
    stages: list[dict[str, Any]] = []
    stage_timings: dict[str, float] = {}
    artifacts: dict[str, str] = {}
    error_msg = ""

    # 用于异常时也写出 pipeline_result
    final_audio_path: str | None = None

    try:
        # ── Stage 1/10: build_segments ──────────────────────────────
        step, name = "1/10", "build_segments"
        _log_stage_start(step, name)
        t0 = time.time()
        text = Path(input_path).read_text(encoding="utf-8-sig")
        story_input = StoryInput(
            story_name=story_name,
            text=text,
            source_path=str(input_path),
        )
        segments_raw = build_segments(story_input)
        elapsed = time.time() - t0
        seg_raw_path = json_dir / "segments_raw.json"
        if save_json:
            _save_json(segments_raw, seg_raw_path)
            artifacts["segments_raw"] = str(seg_raw_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode="run", output=str(seg_raw_path))
        _log_stage_done(step, name, elapsed, extra=f"segments={len(segments_raw)}")

        # ── Stage 2/10: create_llm_client ───────────────────────────
        step, name = "2/10", "create_llm_client"
        _log_stage_start(step, name)
        t0 = time.time()
        llm_backend, llm_cfg = _split_backend_block(profile_dict, "llm")
        llm_client = create_llm_client(llm_backend, **llm_cfg)
        elapsed = time.time() - t0
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode="run", output="")
        # llm_client.model / base_url 可能不存在（mock），用 getattr 兜底
        llm_model = getattr(llm_client, "model", "")
        _log_stage_done(step, name, elapsed,
                        extra=f"backend={llm_backend}, model={llm_model}")

        # ── Stage 3/10: quote_classifier ────────────────────────────
        step, name = "3/10", "quote_classifier"
        _log_stage_start(step, name)
        t0 = time.time()
        merged_path = json_dir / "segments_after_quote_merge.json"
        quote_debug_path = json_dir / "quote_classifications.json"
        mode = "run"
        if reuse:
            reused = _try_load_json(merged_path, _load_segments)
            if reused is not None:
                segments_merged = reused
                mode = "reused"
        if mode == "run":
            segments_merged = classify_and_merge_quotes(
                segments_raw, llm_client,
                story_context=story_name,
                output_debug_path=str(quote_debug_path),
            )
            if save_json:
                _save_json(segments_merged, merged_path)
        elapsed = time.time() - t0
        if save_json:
            artifacts["segments_after_quote_merge"] = str(merged_path)
            if mode == "run" and quote_debug_path.exists():
                artifacts["quote_classifications"] = str(quote_debug_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode=mode, output=str(merged_path))
        if mode == "reused":
            _log_stage_reused(step, name, merged_path.name, elapsed)
        else:
            _log_stage_done(step, name, elapsed,
                            extra=f"segments_after_merge={len(segments_merged)}")

        # ── Stage 4/10: story_resolver ──────────────────────────────
        step, name = "4/10", "story_resolver"
        _log_stage_start(step, name)
        t0 = time.time()
        resolved_path = json_dir / "resolved_segments.json"
        mode = "run"
        if reuse:
            reused = _try_load_json(resolved_path, _load_segments)
            if reused is not None:
                resolved = reused
                mode = "reused"
        if mode == "run":
            resolved = resolve_speakers(segments_merged, llm_client, story_context=story_name)
            if save_json:
                _save_json(resolved, resolved_path)
        elapsed = time.time() - t0
        if save_json:
            artifacts["resolved_segments"] = str(resolved_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode=mode, output=str(resolved_path))
        if mode == "reused":
            _log_stage_reused(step, name, resolved_path.name, elapsed)
        else:
            _log_stage_done(step, name, elapsed, extra=f"resolved={len(resolved)}")

        # ── Stage 5/10: character_analyzer ──────────────────────────
        step, name = "5/10", "character_analyzer"
        _log_stage_start(step, name)
        t0 = time.time()
        characters_path = json_dir / "characters.json"
        mode = "run"
        if reuse:
            reused = _try_load_json(characters_path, _load_characters)
            if reused is not None:
                characters = reused
                mode = "reused"
        if mode == "run":
            characters = analyze_characters(resolved, llm_client, story_context=story_name)
            if save_json:
                _save_json(characters, characters_path)
        elapsed = time.time() - t0
        if save_json:
            artifacts["characters"] = str(characters_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode=mode, output=str(characters_path))
        if mode == "reused":
            _log_stage_reused(step, name, characters_path.name, elapsed)
        else:
            _log_stage_done(step, name, elapsed, extra=f"characters={len(characters)}")

        # ── Stage 6/10: voicebank ───────────────────────────────────
        step, name = "6/10", "voicebank"
        _log_stage_start(step, name)
        t0 = time.time()
        voicebank_result_path = json_dir / "voicebank_result.json"
        mode = "run"
        if reuse:
            reused = _try_load_json(voicebank_result_path, _load_voicebank_result)
            if reused is not None:
                voicebank_result = reused
                mode = "reused"
        if mode == "run":
            vb_backend, vb_cfg = _split_backend_block(profile_dict, "voicebank")
            vb_adapter = create_voicebank_adapter(vb_backend, **vb_cfg)
            voicebank_result = vb_adapter.prepare_voicebank(characters, str(output_dir))
            if save_json:
                _save_json(voicebank_result, voicebank_result_path)
        elapsed = time.time() - t0
        if save_json:
            artifacts["voicebank_result"] = str(voicebank_result_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode=mode,
                              output=str(voicebank_result_path))
        n_voices = len(voicebank_result.speaker_to_voice or {})
        if mode == "reused":
            _log_stage_reused(step, name, voicebank_result_path.name, elapsed)
        else:
            _log_stage_done(step, name, elapsed, extra=f"voices={n_voices}")

        # ── Stage 7/10: story_director ──────────────────────────────
        step, name = "7/10", "story_director"
        _log_stage_start(step, name)
        t0 = time.time()
        director_path = json_dir / "director_plan.json"
        mode = "run"
        if reuse:
            reused = _try_load_json(director_path, _load_director_plan)
            if reused is not None:
                director_plan = reused
                mode = "reused"
        if mode == "run":
            director_plan = generate_director_plan(
                resolved, characters, llm_client, story_context=story_name,
            )
            if save_json:
                _save_json(director_plan, director_path)
        elapsed = time.time() - t0
        if save_json:
            artifacts["director_plan"] = str(director_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode=mode, output=str(director_path))
        if mode == "reused":
            _log_stage_reused(step, name, director_path.name, elapsed)
        else:
            _log_stage_done(step, name, elapsed,
                            extra=f"instructions={len(director_plan)}")

        # ── Stage 8/10: tts_instruction_builder ─────────────────────
        step, name = "8/10", "tts_instruction_builder"
        _log_stage_start(step, name)
        t0 = time.time()
        tts_instructions = build_tts_instructions(
            segments=resolved,
            characters=characters,
            director_plan=director_plan,
            voicebank_result=voicebank_result,
        )
        elapsed = time.time() - t0
        tts_instructions_path = json_dir / "tts_instructions.json"
        if save_json:
            _save_json(tts_instructions, tts_instructions_path)
            artifacts["tts_instructions"] = str(tts_instructions_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status="success",
                              elapsed=elapsed, mode="run",
                              output=str(tts_instructions_path))
        _log_stage_done(step, name, elapsed,
                        extra=f"instructions={len(tts_instructions)}")

        # ── Stage 9/10: tts_synthesis ───────────────────────────────
        step, name = "9/10", "tts_synthesis"
        _log_stage_start(step, name)
        t0 = time.time()

        # 缓存命中检测：所有目标 wav 都已存在且非空 → mode=cached
        audio_subdir = profile_dict["tts"].get("output_subdir", "audio_segments")
        audio_dir = output_dir / audio_subdir
        all_wavs_exist = all(
            (audio_dir / inst.output_filename).exists()
            and (audio_dir / inst.output_filename).stat().st_size > 0
            for inst in tts_instructions
        ) if tts_instructions else False

        tts_backend, tts_cfg = _split_backend_block(profile_dict, "tts")
        tts_adapter = create_tts_adapter(tts_backend, **tts_cfg)
        audio_segments = tts_adapter.synthesize(
            tts_instructions,
            voicebank_result,
            str(output_dir),
            dry_run=False,
            limit=0,
        )
        elapsed = time.time() - t0

        # 不管成败先落盘 audio_segment_results.json（便于排错）
        audio_seg_results_path = json_dir / "audio_segment_results.json"
        _save_json(audio_segments, audio_seg_results_path)
        artifacts["audio_segment_results"] = str(audio_seg_results_path)

        success_n = sum(1 for r in audio_segments if r.success)
        failed_segments = [r for r in audio_segments if not r.success]
        stage_status = "success" if not failed_segments else "failed"
        mode = "cached" if all_wavs_exist else "run"
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name, status=stage_status,
                              elapsed=elapsed, mode=mode,
                              output=str(audio_seg_results_path),
                              error=(f"{len(failed_segments)} segments failed" if failed_segments else None))

        extra_str = f"success={success_n}/{len(audio_segments)}"
        if mode == "cached":
            extra_str += ", cached"
        _log_stage_done(step, name, elapsed, extra=extra_str)

        # stop_on_tts_error 处理（先落盘 stage 9 结果，再抛）
        if failed_segments and stop_on_tts_error:
            err = f"TTS failed {len(failed_segments)}/{len(audio_segments)} segments; stop_on_tts_error=true"
            total_time = time.time() - t_total_start
            summary = _build_pipeline_summary(
                stages=stages, total_time=total_time,
                output_dir=str(output_dir), final_audio_path=None,
            )
            _log_summary(
                success=False, total_time=total_time,
                analysis_time=summary["analysis_time_sec"],
                voicebank_time=summary["voicebank_time_sec"],
                tts_time=summary["tts_time_sec"],
                merge_time=0.0,
                final_audio_duration=None, rtf=None,
                output_dir=str(output_dir), final_audio=None,
                error=err,
            )
            pipeline_result = PipelineResult(
                story_name=story_name,
                output_dir=str(output_dir),
                final_audio=None,
                success=False,
                stage_timings=stage_timings,
                artifacts=artifacts,
                error=err,
                pipeline_summary=summary,
            )
            _save_json(pipeline_result, json_dir / "pipeline_result.json")
            raise RuntimeError(err)

        # ── Stage 10/10: audio_merger ───────────────────────────────
        step, name = "10/10", "audio_merger"
        _log_stage_start(step, name)
        t0 = time.time()
        pause_map = {
            inst.segment_id: inst.pause_hint
            for inst in tts_instructions
            if inst.pause_hint and inst.pause_hint > 0
        }
        final_path = audio_final_dir / f"{story_name}.wav"
        audio_result = merge_audio_segments(
            audio_segments, str(final_path), pause_seconds_after=pause_map,
        )
        elapsed = time.time() - t0
        audio_result_path = json_dir / "audio_result.json"
        if save_json:
            _save_json(audio_result, audio_result_path)
            artifacts["audio_result"] = str(audio_result_path)
        stage_timings[name] = elapsed
        _append_stage_record(stages, name=name,
                              status="success" if audio_result.success else "failed",
                              elapsed=elapsed, mode="run",
                              output=str(audio_result_path))
        rel_final = f"audio_final/{story_name}.wav"
        _log_stage_done(step, name, elapsed, extra=f"final={rel_final}")

        final_audio_path = audio_result.final_audio

        # ── Summary ─────────────────────────────────────────────────
        total_time = time.time() - t_total_start
        success = (not failed_segments) and audio_result.success
        summary = _build_pipeline_summary(
            stages=stages, total_time=total_time,
            output_dir=str(output_dir),
            final_audio_path=final_audio_path,
        )
        _log_summary(
            success=success, total_time=total_time,
            analysis_time=summary["analysis_time_sec"],
            voicebank_time=summary["voicebank_time_sec"],
            tts_time=summary["tts_time_sec"],
            merge_time=summary["merge_time_sec"],
            final_audio_duration=summary["final_audio_duration_sec"],
            rtf=summary["rtf"],
            output_dir=str(output_dir),
            final_audio=final_audio_path,
            error="",
        )

        pipeline_result = PipelineResult(
            story_name=story_name,
            output_dir=str(output_dir),
            final_audio=final_audio_path,
            success=success,
            stage_timings=stage_timings,
            artifacts=artifacts,
            error="",
            pipeline_summary=summary,
        )
        _save_json(pipeline_result, json_dir / "pipeline_result.json")
        artifacts["pipeline_result"] = str(json_dir / "pipeline_result.json")
        return pipeline_result

    except Exception as err:
        # 兜底：尽量把已收集的 stage 信息写到 pipeline_result.json，再返回失败
        error_msg = f"{type(err).__name__}: {err}"
        total_time = time.time() - t_total_start
        summary = _build_pipeline_summary(
            stages=stages, total_time=total_time,
            output_dir=str(output_dir),
            final_audio_path=final_audio_path,
        )
        # 失败前最后一条 stage 日志（如果还没打过）
        # 这里不再尝试 _log_stage_failed，因为可能 stage 已经打过 done 了；
        # 失败信息走 [Summary] 的 error 字段。
        _log_summary(
            success=False, total_time=total_time,
            analysis_time=summary["analysis_time_sec"],
            voicebank_time=summary["voicebank_time_sec"],
            tts_time=summary["tts_time_sec"],
            merge_time=summary["merge_time_sec"],
            final_audio_duration=summary["final_audio_duration_sec"],
            rtf=summary["rtf"],
            output_dir=str(output_dir),
            final_audio=final_audio_path,
            error=error_msg,
        )
        pipeline_result = PipelineResult(
            story_name=story_name,
            output_dir=str(output_dir),
            final_audio=final_audio_path,
            success=False,
            stage_timings=stage_timings,
            artifacts=artifacts,
            error=error_msg,
            pipeline_summary=summary,
        )
        try:
            _save_json(pipeline_result, json_dir / "pipeline_result.json")
        except Exception:
            pass
        return pipeline_result


# ─────────────────────────────────────────────────────────────────────────────
# Mock pipeline（保留不动，给 mock 回归用）
# ─────────────────────────────────────────────────────────────────────────────

def run_mock_core_pipeline(
    input_path: str,
    output_root: str = "output-src-next-core",
) -> PipelineResult:
    """运行最小 mock pipeline，验证 core 层数据流。"""
    story_name = _story_name_from_path(input_path)
    output_dir = Path(output_root) / story_name
    json_dir = output_dir / "json"
    audio_final_dir = output_dir / "audio_final"

    stage_timings: dict = {}
    artifacts: dict = {}

    try:
        # 1. 读 txt + 构造 StoryInput
        text = Path(input_path).read_text(encoding="utf-8-sig")
        story_input = StoryInput(
            story_name=story_name,
            text=text,
            source_path=str(input_path),
        )

        # 2. build_segments
        log_stage_start("1/8", "文本切分")
        t0 = time.time()
        segments = build_segments(story_input)
        stage_timings["build_segments"] = round(time.time() - t0, 3)
        _save_json(segments, json_dir / "segments.json")
        artifacts["segments"] = str(json_dir / "segments.json")
        log_stage_done("1/8", "文本切分")
        log_item(f"生成 {len(segments)} 个 segments")

        # 3. mock characters
        log_stage_start("2/8", "角色档案 (mock)")
        t0 = time.time()
        characters = [
            CharacterProfile(
                name="narrator",
                role_type="narrator",
                gender="neutral",
                age_style="adult",
                personality="旁白",
                voice_prompt="mock narrator voice",
                confidence=1.0,
            )
        ]
        stage_timings["mock_characters"] = round(time.time() - t0, 3)
        _save_json(characters, json_dir / "characters.json")
        artifacts["characters"] = str(json_dir / "characters.json")
        log_stage_done("2/8", "角色档案 (mock)")
        log_item(f"生成 {len(characters)} 个角色档案")

        # 4. mock director_plan
        log_stage_start("3/8", "导演计划 (mock)")
        t0 = time.time()
        director_plan = [
            DirectorInstruction(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                emotion="neutral",
                pace=1.0,
                tone="neutral",
                pause_hint=0.0,
                delivery_instruction="mock delivery",
            )
            for seg in segments
        ]
        stage_timings["mock_director_plan"] = round(time.time() - t0, 3)
        _save_json(director_plan, json_dir / "director_plan.json")
        artifacts["director_plan"] = str(json_dir / "director_plan.json")
        log_stage_done("3/8", "导演计划 (mock)")
        log_item(f"生成 {len(director_plan)} 条导演指导")

        # 5. mock voicebank_result
        log_stage_start("4/8", "Voicebank (mock)")
        t0 = time.time()
        voicebank_result = VoicebankResult(
            speaker_to_voice={"narrator": "mock://voice/narrator"},
            voicebank_dir=str(output_dir / "voicebank"),
            backend="mock",
            success=True,
        )
        stage_timings["mock_voicebank"] = round(time.time() - t0, 3)
        _save_json(voicebank_result, json_dir / "voicebank_result.json")
        artifacts["voicebank_result"] = str(json_dir / "voicebank_result.json")
        log_stage_done("4/8", "Voicebank (mock)")
        log_item(f"准备了 {len(voicebank_result.speaker_to_voice)} 个 voice reference")

        # 6. build_tts_instructions
        log_stage_start("5/8", "TTS 指令构建")
        t0 = time.time()
        tts_instructions = build_tts_instructions(
            segments=segments,
            characters=characters,
            director_plan=director_plan,
            voicebank_result=voicebank_result,
        )
        stage_timings["build_tts_instructions"] = round(time.time() - t0, 3)
        _save_json(tts_instructions, json_dir / "tts_instructions.json")
        artifacts["tts_instructions"] = str(json_dir / "tts_instructions.json")
        log_stage_done("5/8", "TTS 指令构建")
        log_item(f"生成 {len(tts_instructions)} 条 tts 指令")

        # 7. mock audio_segments
        log_stage_start("6/8", "音频合成 (mock)")
        t0 = time.time()
        audio_segments = [
            AudioSegmentResult(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                audio_path=f"mock://audio/{seg.segment_id}",
                success=True,
            )
            for seg in segments
        ]
        stage_timings["mock_audio_segments"] = round(time.time() - t0, 3)
        log_stage_done("6/8", "音频合成 (mock)")
        log_item(f"mock 合成 {len(audio_segments)} 段音频")

        # 8. merge_audio_segments + 占位文件
        log_stage_start("7/8", "音频拼接")
        t0 = time.time()
        audio_final_dir.mkdir(parents=True, exist_ok=True)
        final_audio_path = audio_final_dir / f"{story_name}_mock.txt"
        final_audio_path.write_text(
            f"[MOCK AUDIO PLACEHOLDER]\n"
            f"story: {story_name}\n"
            f"segments: {len(audio_segments)}\n"
            f"generated_by: src_next/core/audiobook_pipeline.py (mock)\n",
            encoding="utf-8",
        )
        audio_result = merge_audio_segments(audio_segments, str(final_audio_path))
        stage_timings["merge_audio"] = round(time.time() - t0, 3)
        _save_json(audio_result, json_dir / "audio_result.json")
        artifacts["audio_result"] = str(json_dir / "audio_result.json")
        log_stage_done("7/8", "音频拼接")
        log_item(f"最终音频: {audio_result.final_audio}")

        # 9. 汇总 pipeline_result
        log_stage_start("8/8", "汇总")
        pipeline_result = PipelineResult(
            story_name=story_name,
            output_dir=str(output_dir),
            final_audio=audio_result.final_audio,
            success=True,
            stage_timings=stage_timings,
            artifacts=artifacts,
            error="",
        )
        _save_json(pipeline_result, json_dir / "pipeline_result.json")
        log_stage_done("8/8", "汇总")
        log_item(f"pipeline success: {pipeline_result.success}")

        return pipeline_result

    except Exception as e:
        return PipelineResult(
            story_name=story_name,
            output_dir=str(output_dir),
            final_audio="",
            success=False,
            stage_timings=stage_timings,
            artifacts=artifacts,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="有声书 Agent 主流程（mock / 真实 pipeline）",
    )
    parser.add_argument("--input", required=True, help="输入 txt 路径")
    parser.add_argument("--profile", default=None,
                        help="profile yaml 路径（--mock 时可省略）")
    parser.add_argument("--output-root", default=None,
                        help="覆盖 profile.output.root；--mock 时覆盖默认 output-src-next-core")
    parser.add_argument("--story-name", default=None,
                        help="覆盖 story_name（默认 input 文件 stem）")
    parser.add_argument("--mock", action="store_true",
                        help="走 run_mock_core_pipeline，忽略 --profile")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="强制 reuse_existing=true（覆盖 profile 设置）")
    return parser.parse_args()


def main() -> int:
    args = _parse_cli_args()

    if args.mock:
        output_root = args.output_root or "output-src-next-core"
        result = run_mock_core_pipeline(args.input, output_root=output_root)
        return 0 if result.success else 1

    if not args.profile:
        print("[ERROR] 真实 pipeline 需要 --profile（或加 --mock 走离线 mock）",
              flush=True)
        return 2

    result = run_pipeline(
        args.input,
        args.profile,
        output_root=args.output_root,
        story_name=args.story_name,
        reuse_existing_override=True if args.reuse_existing else None,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
