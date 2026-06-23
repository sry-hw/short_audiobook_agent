"""src_next/tts/indextts_adapter.py

IndexTTS 后端 adapter（subprocess 调外部 IndexTTS）。

支持两种调用模式：

1. **默认 single 模式**（每条 instruction 跑一次 subprocess）
   * 调 ``python -m indextts.cli <text> --voice ... --output_path ...``
   * 优点：实现简单，失败定位直接看每段 log；
   * 缺点：每条都重新加载模型，慢（IndexTTS 模型加载 ~10-30s）。

2. **batch 模式**（一次 subprocess 跑完整批）
   * 由 ``batch_wrapper_path`` 配置开启，指向
     ``src_next/tts/scripts/run_indextts_batch.py``；
   * adapter 把所有 task 写到 ``batch_input.json``，wrapper 加载一次模型后
     批量合成，输出 ``batch_summary.json``；
   * 优点：整批只加载一次模型，从 ``N × (load + infer)`` 降到
     ``1 × load + N × infer``；
   * 单条失败仍不阻断其他条（wrapper 内部捕获，summary 里标 success=false）。

两种模式都遵守：
* 不硬编码任何 F:/.../indextts 路径，全部通过构造参数传入；
* python_executable 支持 str（直接 python）和 list[str]（wsl 包装）；
* Windows ↔ WSL 路径自动转换（F:\\... → /mnt/f/...）；
* 已存在的 wav 可复用（缓存）；
* 支持 dry_run（不真实调用模型，只写 invocation + style snapshot）；
* 缺 voice_ref 的 instruction 直接 success=False，跳过 subprocess。

IndexTTS 真实接口（来自 ``F:/akoasm/TTS-test/engines/indextts/indextts/cli.py``）：

    python -m indextts.cli <text>
        -v, --voice <wav path>           (required)
        -o, --output_path <wav path>     (default "gen.wav")
        -c, --config <yaml path>         (default "checkpoints/config.yaml")
        --model_dir <dir>                (default "checkpoints")
        --fp16                           (default True)
        -f, --force                      overwrite
        -d, --device <cpu|cuda:0|mps>    (default auto)

**IndexTTS 是纯 zero-shot voice cloning**，**不支持** pace / emotion / volume /
pitch / stress_words / delivery_instruction 等风格参数。这些通用字段在
``TTSInstruction`` 里保留是为了让其它后端（如 CosyVoice2 / Qwen3-TTS）能用；
IndexTTS adapter 把它们写到 per-segment log 留档，**不影响合成结果**。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


_DEFAULT_OUTPUT_SUBDIR = "audio_segments"
_DEFAULT_DEVICE = "cuda:0"
_DEFAULT_CONFIG_REL = "checkpoints/config.yaml"
_DEFAULT_MODEL_DIR_REL = "checkpoints"
_DEFAULT_TIMEOUT_PER_SEG = 600      # single 模式单条超时（含模型加载）
_DEFAULT_BATCH_TIMEOUT_PER_SEG = 120  # batch 模式每条 inference 超时
_DEFAULT_BATCH_MODEL_LOAD_TIMEOUT = 600  # batch 模式模型加载超时

# IndexTTS CLI 入口（用 -m 调，避免依赖 .exe entry point）
_DEFAULT_SCRIPT_MODULE = "indextts.cli"


def _utf8_env() -> dict[str, str]:
    """构造子进程 env：强制 PYTHONIOENCODING=utf-8，避免 Windows cp936 把中文写花。

    子进程 stdout/stderr 被 adapter 重定向到 UTF-8 打开的 .log 文件；如果不
    强制 Python 用 UTF-8 输出，Windows 默认 cp936，中文会变成乱码。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


class IndexTTSAdapter(BaseTTSAdapter):
    """IndexTTS 后端 adapter（支持 single 和 batch 两种模式）。"""

    def __init__(
        self,
        *,
        engine_root: str,
        script_path: str | None = None,
        model_path: str | None = None,
        config_path: str | None = None,
        python_executable: str | list[str] | None = None,
        output_subdir: str = _DEFAULT_OUTPUT_SUBDIR,
        batch_wrapper_path: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        self.engine_root = (engine_root or "").strip()
        # script_path 默认走 -m indextts.cli；single 模式专用
        self.script_path = (script_path or "").strip() or _DEFAULT_SCRIPT_MODULE
        # batch_wrapper_path 不给 → single 模式；给了 → batch 模式
        self.batch_wrapper_path = (batch_wrapper_path or "").strip() or None
        # model_path / config_path 不强制——IndexTTS 默认相对 engine_root 找 checkpoints/
        self.model_path = (model_path or "").strip() or None
        self.config_path = (config_path or "").strip() or None

        # python_executable 接受 str 或 list[str]
        if isinstance(python_executable, (list, tuple)):
            self.python_executable: list[str] = [str(p).strip() for p in python_executable if str(p).strip()]
        else:
            py = (str(python_executable) if python_executable else "").strip()
            self.python_executable = [py] if py else []

        self.output_subdir = (output_subdir or _DEFAULT_OUTPUT_SUBDIR).strip() or _DEFAULT_OUTPUT_SUBDIR
        self.extra_args: dict[str, Any] = dict(extra_args) if extra_args else {}

    # ── BaseTTSAdapter 实现 ──────────────────────────────────────────

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        *,
        dry_run: bool = False,
        timeout_per_seg: int | None = None,
        limit: int = 0,
        **_kwargs: Any,
    ) -> list[AudioSegmentResult]:
        self._validate_config()

        audio_dir = Path(output_dir).expanduser() / self.output_subdir
        audio_dir.mkdir(parents=True, exist_ok=True)

        speaker_to_voice = (voicebank_result.speaker_to_voice if voicebank_result else {}) or {}

        # 落 adapter 配置快照
        self._write_adapter_config(audio_dir, dry_run=dry_run, limit=limit, timeout_per_seg=timeout_per_seg)

        # 分流到 batch / single
        if self.batch_wrapper_path:
            return self._synthesize_batch(
                instructions, speaker_to_voice, audio_dir,
                dry_run=dry_run, limit=limit, timeout_per_seg=timeout_per_seg,
            )
        return self._synthesize_single_loop(
            instructions, speaker_to_voice, audio_dir,
            dry_run=dry_run, limit=limit, timeout_per_seg=timeout_per_seg,
        )

    # ── single 模式 ──────────────────────────────────────────────────

    def _synthesize_single_loop(
        self,
        instructions: list[TTSInstruction],
        speaker_to_voice: dict[str, str],
        audio_dir: Path,
        *,
        dry_run: bool,
        limit: int,
        timeout_per_seg: int | None,
    ) -> list[AudioSegmentResult]:
        n = len(instructions) if not limit or limit <= 0 else min(limit, len(instructions))
        timeout = timeout_per_seg or int(
            self.extra_args.get("timeout_per_seg", _DEFAULT_TIMEOUT_PER_SEG)
        )

        results: list[AudioSegmentResult] = []
        errors: list[str] = []

        for idx in range(len(instructions)):
            inst = instructions[idx]

            # limit 范围外
            if idx >= n:
                results.append(self._skipped_result(inst, "beyond --limit"))
                continue

            # voice_ref 解析
            voice_ref, missing = self._resolve_voice_ref(inst, speaker_to_voice)
            if missing:
                msg = (
                    f"missing voice_ref for speaker={inst.speaker!r}; "
                    "check voicebank_result or tts_instruction_builder fallback chain"
                )
                errors.append(f"{inst.segment_id}: {msg}")
                results.append(self._failed_result(inst, msg))
                continue

            output_wav = audio_dir / inst.output_filename

            # 缓存
            if output_wav.exists() and output_wav.stat().st_size > 0:
                results.append(AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=str(output_wav), success=True, error="",
                ))
                continue

            # dry_run
            if dry_run:
                self._write_dry_run_log_single(inst, output_wav, voice_ref, audio_dir)
                results.append(self._failed_result(inst, "dry_run: not invoked"))
                continue

            # 真实 subprocess
            cmd = self._build_single_command(inst, voice_ref, output_wav)
            log_path = audio_dir / f"{Path(inst.output_filename).stem}.log"

            try:
                with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                    logf.write("=== INVOCATION ===\n")
                    logf.write(" ".join(cmd) + "\n")
                    logf.write(f"cwd={self.engine_root}\n")
                    logf.write(f"text={inst.text!r}\n")
                    logf.write("=== STYLE (NOT passed to IndexTTS; logged only) ===\n")
                    logf.write(self._format_style_snapshot(inst) + "\n")
                    logf.write("=== OUTPUT ===\n")
                    logf.flush()
                    subprocess.run(
                        cmd,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        cwd=self.engine_root or None,
                        timeout=timeout,
                        check=True,
                        env=_utf8_env(),
                    )

                if output_wav.exists() and output_wav.stat().st_size > 0:
                    results.append(AudioSegmentResult(
                        segment_id=inst.segment_id, speaker=inst.speaker,
                        audio_path=str(output_wav), success=True, error="",
                    ))
                else:
                    msg = f"subprocess exit 0 but output wav missing/empty: {output_wav}"
                    errors.append(f"{inst.segment_id}: {msg}")
                    results.append(self._failed_result(inst, msg))
            except subprocess.CalledProcessError as err:
                msg = f"subprocess exit={err.returncode}, see {log_path.name}"
                errors.append(f"{inst.segment_id}: {msg}")
                results.append(self._failed_result(inst, msg))
            except subprocess.TimeoutExpired:
                msg = f"timeout after {timeout}s, see {log_path.name}"
                errors.append(f"{inst.segment_id}: {msg}")
                results.append(self._failed_result(inst, msg))
            except FileNotFoundError as err:
                msg = f"executable not found: {err}. Check python_executable in profile."
                errors.append(f"{inst.segment_id}: {msg}")
                results.append(self._failed_result(inst, msg))
            except Exception as err:  # noqa: BLE001
                msg = f"{type(err).__name__}: {err}"
                errors.append(f"{inst.segment_id}: {msg}")
                results.append(self._failed_result(inst, msg))

        if errors:
            (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
        return results

    # ── batch 模式 ───────────────────────────────────────────────────

    def _synthesize_batch(
        self,
        instructions: list[TTSInstruction],
        speaker_to_voice: dict[str, str],
        audio_dir: Path,
        *,
        dry_run: bool,
        limit: int,
        timeout_per_seg: int | None,
    ) -> list[AudioSegmentResult]:
        """一次 subprocess 合成全部 instructions（模型加载只一次）。"""
        n = len(instructions) if not limit or limit <= 0 else min(limit, len(instructions))

        # 第一遍：分类——哪些直接出结果（limit 外 / 缺 voice_ref / 已缓存 / dry_run），
        # 哪些进 batch_input.json 等真实合成。
        # 用 dict 保留位置：idx → AudioSegmentResult 占位（待 batch 完成后填回）。
        results: list[AudioSegmentResult | None] = [None] * len(instructions)
        tasks: list[dict[str, Any]] = []
        task_indices: list[int] = []  # tasks[i] 对应 instructions 的哪个 idx
        errors: list[str] = []

        for idx in range(len(instructions)):
            inst = instructions[idx]

            if idx >= n:
                results[idx] = self._skipped_result(inst, "beyond --limit")
                continue

            voice_ref, missing = self._resolve_voice_ref(inst, speaker_to_voice)
            if missing:
                msg = (
                    f"missing voice_ref for speaker={inst.speaker!r}; "
                    "check voicebank_result or tts_instruction_builder fallback chain"
                )
                errors.append(f"{inst.segment_id}: {msg}")
                results[idx] = self._failed_result(inst, msg)
                continue

            output_wav = audio_dir / inst.output_filename

            if output_wav.exists() and output_wav.stat().st_size > 0:
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=str(output_wav), success=True, error="",
                )
                continue

            # 进 batch 队列
            tasks.append({
                "segment_id": inst.segment_id,
                "text": inst.text,
                "voice_ref": voice_ref,
                "output_path": str(output_wav),
                "style_text": self._format_style_snapshot(inst),
            })
            task_indices.append(idx)

        # 没有需要真实合成的 task → 直接返回（dry_run 或全缓存）
        if not tasks:
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]

        # 落 batch_input.json
        batch_input_path = audio_dir / "batch_input.json"
        batch_summary_path = audio_dir / "batch_summary.json"
        batch_log_path = audio_dir / "batch.log"

        # cfg_path / model_dir 解析后转 wsl 路径（绝对路径需要转）
        cfg_path = self._resolve_config_path_for_payload()
        model_dir = self._resolve_model_dir_for_payload()

        payload = {
            "config": {
                "cfg_path": cfg_path,
                "model_dir": model_dir,
                "device": str(self.extra_args.get("device", _DEFAULT_DEVICE)),
                "is_fp16": not bool(self.extra_args.get("disable_fp16", False)),
            },
            "tasks": tasks,
        }
        batch_input_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # dry_run：不真实调用 wrapper，写 batch.log 说明
        if dry_run:
            cmd_preview = self._build_batch_command(batch_input_path, batch_summary_path)
            with open(batch_log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write("=== DRY RUN (batch) ===\n")
                logf.write(" ".join(cmd_preview) + "\n")
                logf.write(f"cwd={self.engine_root}\n")
                logf.write(f"input={batch_input_path}\n")
                logf.write(f"summary(target)={batch_summary_path}\n")
                logf.write(f"tasks={len(tasks)}\n")
            # 所有 queued task 标 dry_run
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], "dry_run: batch not invoked")
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]

        # 真实调用 wrapper
        cmd = self._build_batch_command(batch_input_path, batch_summary_path)
        per_seg_timeout = timeout_per_seg or int(
            self.extra_args.get("batch_timeout_per_seg", _DEFAULT_BATCH_TIMEOUT_PER_SEG)
        )
        model_load_timeout = int(
            self.extra_args.get("batch_model_load_timeout", _DEFAULT_BATCH_MODEL_LOAD_TIMEOUT)
        )
        total_timeout = model_load_timeout + per_seg_timeout * len(tasks)

        try:
            with open(batch_log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write("=== INVOCATION (batch) ===\n")
                logf.write(" ".join(cmd) + "\n")
                logf.write(f"cwd={self.engine_root}\n")
                logf.write(f"tasks={len(tasks)}\n")
                logf.write(f"timeout={total_timeout}s (model_load={model_load_timeout} + "
                           f"{len(tasks)} × {per_seg_timeout})\n")
                logf.write("=== OUTPUT ===\n")
                logf.flush()
                subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=self.engine_root or None,
                    timeout=total_timeout,
                    check=True,
                    env=_utf8_env(),
                )
        except subprocess.CalledProcessError as err:
            msg = f"batch subprocess exit={err.returncode}, see {batch_log_path.name}"
            errors.append(f"batch: {msg}")
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], msg)
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]
        except subprocess.TimeoutExpired:
            msg = f"batch timeout after {total_timeout}s, see {batch_log_path.name}"
            errors.append(f"batch: {msg}")
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], msg)
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]
        except FileNotFoundError as err:
            msg = f"executable not found: {err}. Check python_executable in profile."
            errors.append(f"batch: {msg}")
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], msg)
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]

        # 解析 summary
        if not batch_summary_path.exists():
            msg = f"batch finished but summary missing: {batch_summary_path.name}"
            errors.append(f"batch: {msg}")
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], msg)
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]

        try:
            summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        except Exception as err:  # noqa: BLE001
            msg = f"failed to parse summary: {type(err).__name__}: {err}"
            errors.append(f"batch: {msg}")
            for idx in task_indices:
                results[idx] = self._failed_result(instructions[idx], msg)
            if errors:
                (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
            return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                    for i, r in enumerate(results)]

        # 把 summary 的 results 按 segment_id 映射回原位置
        # summary["results"] 是 wrapper 内部顺序（tasks 顺序，和 task_indices 对应）
        summary_results: list[dict[str, Any]] = summary.get("results", [])
        # 兜底：模型加载失败时 wrapper 也会给每个 task 写一个 fail 条目
        # 但如果 wrapper 提前 crash 没有 summary，已经 above 处理
        seg_id_to_summary: dict[str, dict[str, Any]] = {
            r.get("segment_id", ""): r for r in summary_results
        }

        for idx in task_indices:
            inst = instructions[idx]
            r = seg_id_to_summary.get(inst.segment_id)
            if r is None:
                results[idx] = self._failed_result(inst, "missing from batch summary")
                errors.append(f"{inst.segment_id}: missing from batch summary")
                continue
            success = bool(r.get("success"))
            audio_path = r.get("output_path") or str(audio_dir / inst.output_filename)
            # 再校验文件确实存在
            if success and not (Path(audio_path).exists() and Path(audio_path).stat().st_size > 0):
                success = False
                err_msg = "batch summary marked success but wav missing/empty"
            else:
                err_msg = r.get("error", "") or ""
            results[idx] = AudioSegmentResult(
                segment_id=inst.segment_id,
                speaker=inst.speaker,
                audio_path=audio_path if success else None,
                success=success,
                error=err_msg if not success else "",
            )
            if not success:
                errors.append(f"{inst.segment_id}: {err_msg}")

        if errors:
            (audio_dir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")

        # 真正落盘前已经把 None 全替换掉了，统一兜底一遍
        return [r if r is not None else self._failed_result(instructions[i], "internal: result not set")
                for i, r in enumerate(results)]

    # ── 内部工具 ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        if not self.engine_root:
            raise TTSError(
                "IndexTTSAdapter 缺少 engine_root。"
                "请在 profile 中配置 tts.engine_root 指向 indextts 仓库根目录。"
            )

    def _resolve_voice_ref(
        self,
        inst: TTSInstruction,
        speaker_to_voice: dict[str, str],
    ) -> tuple[str, bool]:
        v = (inst.voice_ref or "").strip()
        if v:
            return v, False
        v = (speaker_to_voice.get(inst.speaker) or "").strip()
        if v:
            return v, False
        return "", True

    def _is_wsl_invocation(self) -> bool:
        if not self.python_executable:
            return False
        first = self.python_executable[0].lower()
        return first == "wsl" or first.endswith("wsl.exe") or first.endswith("\\wsl")

    def _to_subprocess_path(self, p: str | Path) -> str:
        """Windows → WSL 路径转换（仅当 python_executable 是 wsl 包装时）。"""
        s = str(p).replace("\\", "/")
        if not self._is_wsl_invocation():
            return s
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            rest = s[2:].lstrip("/")
            return f"/mnt/{drive}/{rest}"
        return s

    def _python_command_parts(self) -> list[str]:
        if self.python_executable:
            return list(self.python_executable)
        return [sys.executable]

    @staticmethod
    def _skipped_result(inst: TTSInstruction, reason: str) -> AudioSegmentResult:
        return AudioSegmentResult(
            segment_id=inst.segment_id, speaker=inst.speaker,
            audio_path=None, success=False, error=f"skipped: {reason}",
        )

    @staticmethod
    def _failed_result(inst: TTSInstruction, msg: str) -> AudioSegmentResult:
        return AudioSegmentResult(
            segment_id=inst.segment_id, speaker=inst.speaker,
            audio_path=None, success=False, error=msg,
        )

    def _write_adapter_config(
        self,
        audio_dir: Path,
        *,
        dry_run: bool,
        limit: int,
        timeout_per_seg: int | None,
    ) -> None:
        config_snapshot = {
            "engine_root": self.engine_root,
            "mode": "batch" if self.batch_wrapper_path else "single",
            "script_path": self.script_path,
            "batch_wrapper_path": self.batch_wrapper_path,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "python_executable": self.python_executable,
            "output_subdir": self.output_subdir,
            "extra_args": self.extra_args,
            "dry_run": dry_run,
            "limit": limit,
            "timeout_per_seg": timeout_per_seg,
        }
        (audio_dir / "adapter_config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _format_style_snapshot(self, inst: TTSInstruction) -> str:
        stress = inst.stress_words if isinstance(inst.stress_words, list) else []
        return (
            f"emotion={inst.emotion}; intensity={inst.emotion_intensity:.2f}; "
            f"tone={inst.tone}; volume={inst.volume}; pitch={inst.pitch}; "
            f"pace={inst.pace:.2f}; pause_hint={inst.pause_hint:.2f}; "
            f"stress_words={stress}; delivery={inst.delivery_instruction!r}"
        )

    # ── single 模式专属 ──────────────────────────────────────────────

    def _is_module_invocation(self) -> bool:
        s = self.script_path.replace("\\", "/")
        if s.endswith(".py"):
            return False
        return "/" not in s

    def _resolve_script_path(self) -> str:
        if self._is_module_invocation():
            return self.script_path
        script = Path(self.script_path)
        if script.is_absolute():
            return str(script)
        return str(Path(self.engine_root) / script)

    def _resolve_config_path(self) -> str:
        if self.config_path:
            cfg = Path(self.config_path)
            if cfg.is_absolute():
                return str(cfg)
            return str(Path(self.engine_root) / cfg)
        return _DEFAULT_CONFIG_REL

    def _resolve_model_dir(self) -> str:
        if self.model_path:
            md = Path(self.model_path)
            if md.is_absolute():
                return str(md)
            return str(Path(self.engine_root) / md)
        return _DEFAULT_MODEL_DIR_REL

    def _build_single_command(
        self,
        inst: TTSInstruction,
        voice_ref: str,
        output_wav: Path,
    ) -> list[str]:
        py_parts = self._python_command_parts()
        script = self._to_subprocess_path(self._resolve_script_path())

        cmd: list[str] = list(py_parts)
        if self._is_module_invocation():
            cmd += ["-m", script]
        else:
            cmd += [script]

        cmd += [inst.text]
        cmd += ["--voice", self._to_subprocess_path(voice_ref)]
        cmd += ["--output_path", self._to_subprocess_path(output_wav)]

        cfg = self._resolve_config_path()
        mdir = self._resolve_model_dir()
        if self._is_wsl_invocation() and (Path(cfg).is_absolute() or Path(mdir).is_absolute()):
            cmd += ["--config", self._to_subprocess_path(cfg)]
            cmd += ["--model_dir", self._to_subprocess_path(mdir)]
        else:
            cmd += ["--config", cfg]
            cmd += ["--model_dir", mdir]

        cmd += ["--device", str(self.extra_args.get("device", _DEFAULT_DEVICE))]
        if not self.extra_args.get("disable_fp16", False):
            cmd += ["--fp16"]
        cmd += ["--force"]
        return cmd

    def _write_dry_run_log_single(
        self,
        inst: TTSInstruction,
        output_wav: Path,
        voice_ref: str,
        audio_dir: Path,
    ) -> None:
        log_path = audio_dir / f"{Path(inst.output_filename).stem}.log"
        cmd_preview = self._build_single_command(inst, voice_ref, output_wav)
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            logf.write("=== DRY RUN ===\n")
            logf.write(" ".join(cmd_preview) + "\n")
            logf.write(f"cwd={self.engine_root}\n")
            logf.write(f"text={inst.text!r}\n")
            logf.write(f"voice_ref={voice_ref}\n")
            logf.write(f"output={output_wav}\n")
            logf.write("=== STYLE (NOT passed to IndexTTS; logged only) ===\n")
            logf.write(self._format_style_snapshot(inst) + "\n")

    # ── batch 模式专属 ───────────────────────────────────────────────

    def _resolve_wrapper_script_path(self) -> str:
        """batch_wrapper_path 解析：绝对路径原样返回，相对路径相对 engine_root。"""
        w = Path(self.batch_wrapper_path or "")
        if w.is_absolute():
            return str(w)
        return str(Path(self.engine_root) / w)

    def _resolve_config_path_for_payload(self) -> str:
        """batch_input.json 里的 cfg_path：传给 wrapper 内部解析。

        wrapper 自己会按 --cwd 解析相对路径；adapter 这里给绝对路径或相对 engine_root
        都行（wrapper 的 --cwd 默认 = engine_root）。
        """
        if self.config_path:
            cfg = Path(self.config_path)
            if cfg.is_absolute():
                return self._to_subprocess_path(cfg)
            return self._to_subprocess_path(Path(self.engine_root) / cfg)
        # 默认相对路径，wrapper 端按 cwd 解析
        return _DEFAULT_CONFIG_REL

    def _resolve_model_dir_for_payload(self) -> str:
        if self.model_path:
            md = Path(self.model_path)
            if md.is_absolute():
                return self._to_subprocess_path(md)
            return self._to_subprocess_path(Path(self.engine_root) / md)
        return _DEFAULT_MODEL_DIR_REL

    def _build_batch_command(
        self,
        batch_input_path: Path,
        batch_summary_path: Path,
    ) -> list[str]:
        py_parts = self._python_command_parts()
        wrapper = self._to_subprocess_path(self._resolve_wrapper_script_path())
        cmd = list(py_parts) + [wrapper]
        cmd += ["--input", self._to_subprocess_path(batch_input_path)]
        cmd += ["--summary", self._to_subprocess_path(batch_summary_path)]
        if self.engine_root:
            cmd += ["--cwd", self._to_subprocess_path(self.engine_root)]
        return cmd
