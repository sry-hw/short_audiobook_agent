"""src_next/tts/scripts/run_indextts_batch.py

IndexTTS 批量合成 wrapper：启动时加载一次模型，批量合成多个 segment。

被 ``src_next/tts/indextts_adapter.py`` 在 batch 模式下调用（profile 配置
``extra_args.batch_wrapper_path`` 时启用）。和默认的"每条 segment 跑一次
subprocess"模式相比，省掉 N-1 次模型加载（IndexTTS ~10-30s/次），整批
从 ``N × (load + infer)`` 降到 ``1 × load + N × infer``。

设计要点：
* 输入：JSON 文件（config + tasks[]）。adapter 把所有 segment 写进去；
* 输出：JSON summary 文件，per-task success/error 写明；
* 单条失败不阻断 loop；
* 缓存：output wav 已存在且非空 → 跳过 infer，标记 cached=True；
* 模型加载失败立即退出（exit code 2），整体失败由 adapter 用 summary 处理；
* IndexTTS 是 zero-shot voice cloning，没有 emotion / pace / volume 等参数；
  tasks 里的 ``style_text`` 字段只用于日志，不传给 infer。

CLI：
    python run_indextts_batch.py \\
        --input <batch input json> \\
        --summary <batch summary json> \\
        [--cwd <engine_root，仅用于 cfg_path/model_dir 相对路径解析>]

不接 CLI 的 cfg_path / model_dir / device——这些走 input json 的 ``config``
字段，避免 adapter 拼一堆 --flag 时漏掉某个。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IndexTTS batch synthesis wrapper (load model once)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Batch input JSON path (config + tasks)",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Batch summary JSON output path (per-task results)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Engine root for resolving relative cfg_path / model_dir (default: cwd)",
    )
    return parser.parse_args()


def _resolve_path(p: str, cwd: str | None) -> str:
    """相对路径按 cwd 解析；绝对路径原样返回。"""
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    if cwd:
        return str(Path(cwd) / pp)
    return str(pp.resolve())


def _load_indextts(config: dict[str, Any], cwd: str | None) -> Any:
    """加载 IndexTTS 模型（只加载一次）。"""
    from indextts.infer import IndexTTS  # 懒导入，避免 --help 时拉依赖

    cfg_path = _resolve_path(
        config.get("cfg_path") or "checkpoints/config.yaml", cwd
    )
    model_dir = _resolve_path(
        config.get("model_dir") or "checkpoints", cwd
    )
    device = config.get("device") or "cuda:0"
    is_fp16 = bool(config.get("is_fp16", True))

    # 简单存在性校验，给出可读错误
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"cfg_path 不存在: {cfg_path}")
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir 不存在: {model_dir}")

    print(f"[indexTTS] cfg_path   = {cfg_path}", flush=True)
    print(f"[indexTTS] model_dir  = {model_dir}", flush=True)
    print(f"[indexTTS] device     = {device}", flush=True)
    print(f"[indexTTS] is_fp16    = {is_fp16}", flush=True)

    return IndexTTS(
        cfg_path=cfg_path,
        model_dir=model_dir,
        is_fp16=is_fp16,
        device=device,
    )


def _synthesize_one(tts: Any, task: dict[str, Any]) -> dict[str, Any]:
    """合成单条 task；不抛异常，错误写进返回字典。"""
    seg_id = task.get("segment_id") or "<unknown>"
    text = task.get("text") or ""
    voice_ref = task.get("voice_ref") or ""
    output_path = task.get("output_path") or ""

    result: dict[str, Any] = {
        "segment_id": seg_id,
        "output_path": output_path,
        "success": False,
        "error": "",
        "duration_seconds": 0.0,
        "cached": False,
    }

    # 基本校验
    if not text.strip():
        result["error"] = "empty text"
        return result
    if not voice_ref or not Path(voice_ref).exists():
        result["error"] = f"voice_ref missing: {voice_ref}"
        return result
    if not output_path:
        result["error"] = "output_path empty"
        return result

    # 缓存：已存在且非空 → 直接跳过
    out_p = Path(output_path)
    if out_p.exists() and out_p.stat().st_size > 0:
        result["success"] = True
        result["cached"] = True
        return result

    # 真实合成
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        # IndexTTS infer 签名（来自 indextts/infer.py）：
        #   tts.infer(audio_prompt, text, output_path)
        tts.infer(
            audio_prompt=voice_ref,
            text=text,
            output_path=output_path,
        )
        elapsed = time.time() - t0
        result["duration_seconds"] = round(elapsed, 3)
        # 校验产物
        if out_p.exists() and out_p.stat().st_size > 0:
            result["success"] = True
        else:
            result["error"] = "infer returned but wav missing/empty"
    except Exception as err:  # noqa: BLE001
        elapsed = time.time() - t0
        result["duration_seconds"] = round(elapsed, 3)
        result["error"] = f"{type(err).__name__}: {err}"
    return result


def main() -> int:
    args = _parse_args()
    cwd = args.cwd or None

    # 1. 读 input JSON
    input_path = Path(args.input).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] input JSON 不存在: {input_path}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as err:  # noqa: BLE001
        print(f"[ERROR] 解析 input JSON 失败: {err}", file=sys.stderr)
        return 2

    config = payload.get("config") or {}
    tasks = payload.get("tasks") or []
    if not tasks:
        print("[WARN] tasks 为空，写空 summary 后退出")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "backend": "indextts_batch",
                    "success": False,
                    "error": "no tasks",
                    "results": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 加载模型（只一次）
    print(f"[indexTTS] loading model... ({len(tasks)} tasks)", flush=True)
    t_load_start = time.time()
    try:
        tts = _load_indextts(config, cwd)
    except Exception as err:  # noqa: BLE001
        err_msg = f"model load failed: {type(err).__name__}: {err}"
        print(f"[ERROR] {err_msg}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        summary_path.write_text(
            json.dumps(
                {
                    "backend": "indextts_batch",
                    "success": False,
                    "error": err_msg,
                    "results": [
                        {
                            "segment_id": t.get("segment_id", ""),
                            "output_path": t.get("output_path", ""),
                            "success": False,
                            "error": "model load failed; not attempted",
                            "duration_seconds": 0.0,
                            "cached": False,
                        }
                        for t in tasks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 3
    load_seconds = time.time() - t_load_start
    print(f"[indexTTS] model loaded in {load_seconds:.2f}s", flush=True)

    # 3. 逐条合成
    t_total_start = time.time()
    results: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        seg_id = task.get("segment_id") or f"task_{idx}"
        style_text = task.get("style_text") or ""
        print(f"\n[{seg_id}] START ({idx + 1}/{len(tasks)})", flush=True)
        if style_text:
            # IndexTTS 不消费 style，只留档
            print(f"[{seg_id}] STYLE (not passed to IndexTTS): {style_text}", flush=True)
        print(f"[{seg_id}] text={task.get('text', '')[:80]!r}", flush=True)
        print(f"[{seg_id}] voice_ref={task.get('voice_ref')}", flush=True)
        print(f"[{seg_id}] output={task.get('output_path')}", flush=True)

        r = _synthesize_one(tts, task)
        results.append(r)

        status = "OK" if r["success"] else "FAIL"
        cached_tag = " (cached)" if r.get("cached") else ""
        err_tag = f" — {r['error']}" if not r["success"] else ""
        print(
            f"[{seg_id}] {status}{cached_tag} ({r['duration_seconds']:.2f}s){err_tag}",
            flush=True,
        )

    total_seconds = time.time() - t_total_start
    success_count = sum(1 for r in results if r["success"])

    # 4. 写 summary
    summary = {
        "backend": "indextts_batch",
        "success": success_count > 0,
        "model_load_seconds": round(load_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "synthesize_seconds": round(total_seconds - 0, 3),
        "task_count": len(tasks),
        "success_count": success_count,
        "fail_count": len(tasks) - success_count,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n[indexTTS] done: {success_count}/{len(tasks)} ok, "
        f"model_load={load_seconds:.2f}s, total={total_seconds:.2f}s",
        flush=True,
    )
    print(f"[indexTTS] summary: {summary_path}", flush=True)
    return 0 if success_count > 0 else 4


if __name__ == "__main__":
    sys.exit(main())
