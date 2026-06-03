"""TTS 引擎：读取指令列表，调用 MOSS-TTS-Nano ONNX Runtime 生成音频文件，并拼接为完整 wav。"""

import sys
import time
import wave
from pathlib import Path
from typing import Dict, List

_MOSS_PATH = "M:/Users/l30083418/Documents/MOSS-TTS-Nano"
_MODEL_DIR = "M:/Users/l30083418/Documents/MOSS-TTS-Nano/models"

_runtime = None


def synthesize_all(
    instructions: Dict, output_dir: str = "output/audio_segments"
) -> Dict:
    """遍历 instructions 逐条合成语音。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    runtime = _get_runtime()
    results = []
    success = 0
    failed = 0

    for i, inst in enumerate(instructions["instructions"]):
        text_preview = inst["text"][:30]
        print(f"[{i + 1}/{instructions['total_instructions']}] {inst['speaker']}: {text_preview}...", end=" ", flush=True)
        result = _synthesize_one(runtime, inst, output_dir)
        results.append(result)

        if result["status"] == "success":
            success += 1
            print(f"完成 ({result['elapsed_seconds']:.1f}s)")
        else:
            failed += 1
            print(f"失败: {result['error']}")

    return {
        "results": results,
        "total": len(results),
        "success": success,
        "failed": failed,
    }


def stitch_audio(
    results: List[Dict],
    output_path: str = "output/audio_final/final.wav",
    pause_seconds: float = 0.5,
) -> Dict:
    """按 segment_id 顺序拼接所有成功生成的 wav 文件。

    Args:
        results: synthesize_all 返回的 results 列表
        output_path: 最终输出 wav 路径
        pause_seconds: 段间静音时长（秒）

    Returns:
        包含输出路径和时长信息的字典
    """
    successful = [r for r in results if r["status"] == "success"]
    successful.sort(key=lambda r: r["segment_id"])

    if not successful:
        return {"output_path": "", "total_duration_seconds": 0, "total_segments": 0}

    # 读取第一个文件获取音频参数
    first = successful[0]["audio_path"]
    with wave.open(first, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

    pause_frames = int(framerate * pause_seconds)
    pause_bytes = b"\x00" * (pause_frames * channels * sampwidth)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with wave.open(output_path, "wb") as out_wf:
        out_wf.setnchannels(channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)

        for i, r in enumerate(successful):
            if not r["audio_path"] or not Path(r["audio_path"]).exists():
                continue
            with wave.open(r["audio_path"], "rb") as seg_wf:
                out_wf.writeframes(seg_wf.readframes(seg_wf.getnframes()))
            # 段间停顿（最后一段不加）
            if i < len(successful) - 1:
                out_wf.writeframes(pause_bytes)

    total_frames = wave.open(output_path, "rb").getnframes()
    duration = round(total_frames / framerate, 2)

    return {
        "output_path": str(Path(output_path).resolve()),
        "total_duration_seconds": duration,
        "total_segments": len(successful),
    }


def synthesize_and_stitch(
    instructions: Dict,
    output_dir: str = "output/audio_segments",
    final_output: str = "output/audio_final/final.wav",
) -> Dict:
    """合成所有片段并拼接为完整音频。"""
    synth_result = synthesize_all(instructions, output_dir)
    stitch_result = stitch_audio(synth_result["results"], final_output)

    return {
        "synthesis": synth_result,
        "stitch": stitch_result,
    }


def _get_runtime():
    """懒加载 OnnxTtsRuntime，只初始化一次。"""
    global _runtime
    if _runtime is not None:
        return _runtime

    if _MOSS_PATH not in sys.path:
        sys.path.insert(0, _MOSS_PATH)

    from onnx_tts_runtime import OnnxTtsRuntime

    print("正在加载 MOSS-TTS-Nano ONNX 模型...")
    _runtime = OnnxTtsRuntime(model_dir=_MODEL_DIR)
    print("模型加载完成")
    return _runtime


def _synthesize_one(runtime, instruction: Dict, output_dir: str) -> Dict:
    """合成单条指令。"""
    try:
        output_path = str(Path(output_dir) / instruction["output_filename"])
        params = instruction["tts_params"]

        t0 = time.perf_counter()
        runtime.synthesize(
            text=instruction["text"],
            prompt_audio_path=instruction["prompt_audio_path"],
            do_sample=params.get("do_sample", True),
            output_audio_path=output_path,
        )
        elapsed = time.perf_counter() - t0

        return {
            "segment_id": instruction["segment_id"],
            "speaker": instruction["speaker"],
            "audio_path": output_path,
            "elapsed_seconds": round(elapsed, 2),
            "status": "success",
        }
    except Exception as e:
        return {
            "segment_id": instruction["segment_id"],
            "speaker": instruction["speaker"],
            "audio_path": "",
            "elapsed_seconds": 0,
            "status": "failed",
            "error": str(e),
        }
