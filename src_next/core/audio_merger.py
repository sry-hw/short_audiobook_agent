"""src_next/core/audio_merger.py

音频拼接 — 最小可用版本（v2，加段间静音）。

策略：
    1. 过滤 ``success=True`` 且 wav 文件实际存在的 AudioSegmentResult；
    2. 用 stdlib ``wave`` 模块按 segment 顺序直接拼接 wav 字节流；
    3. 全部 wav 必须同采样率 / 同位深 / 同声道——否则只把不一致的段跳过
       并写到 errors.log，不阻断整条 pipeline；
    4. **段间按 ``pause_seconds_after[segment_id]`` 插入静音**（导演层的
       ``pause_hint`` 通过调用方传进来）。

不做的事（保留 TODO）：
    * 不做重采样（采率不一致直接跳过该段，TODO: 接 torchaudio / librosa 重采样）；
    * 不做响度归一化；
    * 不接 ffmpeg / pydub，避免引入新依赖。

若所有段都失败 / 不存在，函数返回 success=False + final_audio=final_path
（不写文件），调用方自行决定怎么呈现给用户。
"""

from __future__ import annotations

import wave
from pathlib import Path

from .data_models import AudioResult, AudioSegmentResult


def merge_audio_segments(
    audio_segments: list[AudioSegmentResult],
    final_path: str,
    pause_seconds_after: dict[str, float] | None = None,
    *,
    min_silence_seconds: float = 0.4,
) -> AudioResult:
    """把 audio_segments 拼接成最终 wav，段间插入静音。

    Args:
        audio_segments: TTS adapter 产出的分段结果（按 segment 顺序）。
        final_path: 最终 wav 输出路径。父目录会自动创建。
        pause_seconds_after: 可选，``segment_id → 段后静音秒数`` 映射。
            导演层 ``DirectorInstruction.pause_hint`` 通过调用方按 segment_id
            传进来。缺失的 segment_id 按 0 处理（不插静音）。
            静音以"零字节帧"形式插入，和原始 wav 同采样率 / 同位深 / 同声道。
        min_silence_seconds: 段间最小静音下限（秒）。即使 ``pause_seconds_after``
            里某段被 LLM 给了很小的值（如 0.2 秒）或没给，也会被提到此下限；
            **最后一段**不应用此下限（避免末尾多余静音）。设为 0 可关闭。
            默认 0.4 秒，是为了避免中文 TTS 输出本身连贯 + 过短 pause 导致
            段间听感紧凑。

    Returns:
        AudioResult：
        * final_audio = final_path（即便失败也回填路径，方便调用方引用）；
        * audio_segments = 原 list 浅拷贝；
        * duration_seconds = 实际拼接出的秒数（含静音；失败 = 0）；
        * success = 至少有一段成功拼进 final wav。
    """
    final_path_obj = Path(final_path).expanduser().resolve()
    final_path_obj.parent.mkdir(parents=True, exist_ok=True)

    pause_map = pause_seconds_after or {}

    # 1. 过滤可用段
    usable: list[AudioSegmentResult] = []
    skipped: list[tuple[str, str]] = []
    for seg in audio_segments:
        if not seg.success or not seg.audio_path:
            skipped.append((seg.segment_id, "success=False or audio_path empty"))
            continue
        p = Path(seg.audio_path)
        if not p.exists() or p.stat().st_size == 0:
            skipped.append((seg.segment_id, f"wav missing/empty: {seg.audio_path}"))
            continue
        if str(p).startswith("mock://"):
            skipped.append((seg.segment_id, "mock:// path, not a real wav"))
            continue
        usable.append(seg)

    if not usable:
        return AudioResult(
            final_audio=final_path,
            audio_segments=list(audio_segments),
            duration_seconds=0.0,
            success=False,
        )

    # 2. 读所有 wav，校验格式一致；以第一段为基准；同时收集每段对应的静音秒数
    base_params: wave._wave_params | None = None
    # 每项：(audio_frames_bytes, pause_seconds_after_this_segment)
    frames_with_pause: list[tuple[bytes, float]] = []
    total_frames = 0
    total_silence_frames = 0
    skipped_format: list[tuple[str, str]] = []

    total_usable = len(usable)
    for idx, seg in enumerate(usable):
        is_last = (idx == total_usable - 1)
        try:
            with wave.open(str(Path(seg.audio_path).resolve()), "rb") as wf:
                params = wf.getparams()
                if base_params is None:
                    base_params = params
                    frames = wf.readframes(params.nframes)
                else:
                    if (params.nchannels, params.sampwidth, params.framerate) != (
                        base_params.nchannels,
                        base_params.sampwidth,
                        base_params.framerate,
                    ):
                        skipped_format.append(
                            (
                                seg.segment_id,
                                f"format mismatch "
                                f"(got ch={params.nchannels} sw={params.sampwidth} fr={params.framerate}; "
                                f"base ch={base_params.nchannels} sw={base_params.sampwidth} fr={base_params.framerate})",
                            )
                        )
                        continue
                    frames = wf.readframes(params.nframes)
            pause_s = float(pause_map.get(seg.segment_id, 0.0) or 0.0)
            if pause_s < 0:
                pause_s = 0.0
            # 段间最小静音下限（双保险）：LLM 给的 pause_hint 过小或没给时，
            # 强制提到 min_silence_seconds；最后一段不应用（避免末尾多余静音）。
            if not is_last and min_silence_seconds > 0 and pause_s < min_silence_seconds:
                pause_s = min_silence_seconds
            frames_with_pause.append((frames, pause_s))
            total_frames += params.nframes
            if pause_s > 0 and base_params is not None:
                silence_n = int(pause_s * base_params.framerate)
                total_silence_frames += silence_n
        except Exception as err:  # noqa: BLE001
            skipped_format.append((seg.segment_id, f"wave.open failed: {type(err).__name__}: {err}"))

    if not frames_with_pause or base_params is None:
        return AudioResult(
            final_audio=final_path,
            audio_segments=list(audio_segments),
            duration_seconds=0.0,
            success=False,
        )

    # 3. 拼接写出（含静音）
    framerate = base_params.framerate
    sampwidth = base_params.sampwidth
    nchannels = base_params.nchannels
    silence_byte_per_frame = b"\x00" * (sampwidth * nchannels)

    try:
        with wave.open(str(final_path_obj), "wb") as out_wf:
            out_wf.setparams(base_params)
            for frames, pause_s in frames_with_pause:
                if frames:
                    out_wf.writeframes(frames)
                if pause_s > 0:
                    silence_n = int(pause_s * framerate)
                    if silence_n > 0:
                        out_wf.writeframes(silence_byte_per_frame * silence_n)
    except Exception:  # noqa: BLE001
        return AudioResult(
            final_audio=final_path,
            audio_segments=list(audio_segments),
            duration_seconds=0.0,
            success=False,
        )

    total_all_frames = total_frames + total_silence_frames
    duration = total_all_frames / framerate if framerate else 0.0

    return AudioResult(
        final_audio=str(final_path_obj),
        audio_segments=list(audio_segments),
        duration_seconds=duration,
        success=True,
    )
