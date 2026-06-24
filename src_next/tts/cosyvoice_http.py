"""src_next/tts/cosyvoice_http.py

Fun-CosyVoice3-0.5B HTTP TTS adapter（黄区内网调用，并发版）。

调用方式：HTTP POST 到 ``{base_url}/v1/cosyvoice/generate``，请求体包含
``{text, prompt_text, prompt_audio, mode, stream}``，响应体直接是 wav 字节流。

API 参考：``usage_guide_cosyvoice.md``（项目根目录）。

参数映射（instruct 模式，最常用）：
    instruction.text              → payload.text
    instruction.voice_ref         → 读取 wav → base64 → payload.prompt_audio
    instruction.emotion + tone + volume + pace + delivery_instruction
                                  → 拼自然语言 prompt → payload.prompt_text
                                    （末尾自动加 ``.<|endofprompt|>`` 分隔符）

CosyVoice3 比 IndexTTS 表达力更强：自然语言指令可以控制方言 / 语速 / 情绪，
所以通用字段直接合并到 prompt_text，不需要 emotion_vector 这套映射。

并发：默认 4 线程并发合成，可经 ``extra_args.max_workers`` 调整。
"""

from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


urllib3.disable_warnings(InsecureRequestWarning)


_DEFAULT_OUTPUT_SUBDIR = "audio_segments"
_DEFAULT_MODE = "instruct"
_DEFAULT_TIMEOUT_PER_SEG = 60
_DEFAULT_MAX_WORKERS = 4
_ENDOFPROMPT = "<|endofprompt|>"


class CosyVoiceHTTPAdapter(BaseTTSAdapter):
    """Fun-CosyVoice3-0.5B 的 HTTP adapter（并发合成）。"""

    def __init__(
        self,
        *,
        base_url: str,
        output_subdir: str = _DEFAULT_OUTPUT_SUBDIR,
        extra_args: dict[str, Any] | None = None,
        **_unused: Any,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise TTSError(
                "CosyVoiceHTTPAdapter 缺少 base_url。"
                "请在 profile 配置 tts.base_url，例如 http://10.50.121.102:8005"
            )
        self.output_subdir = (output_subdir or _DEFAULT_OUTPUT_SUBDIR).strip() or _DEFAULT_OUTPUT_SUBDIR
        self.extra_args: dict[str, Any] = dict(extra_args) if extra_args else {}
        self.bypass_proxy = bool(self.extra_args.get("bypass_proxy", True))
        self.mode = str(self.extra_args.get("mode", _DEFAULT_MODE))

    # ── BaseTTSAdapter 实现 ──────────────────────────────────────────

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        *,
        dry_run: bool = False,
        limit: int = 0,
        max_workers: int | None = None,
        **_kwargs: Any,
    ) -> list[AudioSegmentResult]:
        audio_dir = Path(output_dir).expanduser() / self.output_subdir
        audio_dir.mkdir(parents=True, exist_ok=True)

        speaker_to_voice = (voicebank_result.speaker_to_voice if voicebank_result else {}) or {}

        config_snapshot = {
            "backend": "cosyvoice_http",
            "base_url": self.base_url,
            "output_subdir": self.output_subdir,
            "extra_args": self.extra_args,
            "mode": self.mode,
            "dry_run": dry_run,
            "limit": limit,
            "max_workers": max_workers or int(self.extra_args.get("max_workers", _DEFAULT_MAX_WORKERS)),
        }
        (audio_dir / "adapter_config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        n = len(instructions) if not limit or limit <= 0 else min(limit, len(instructions))
        timeout = int(self.extra_args.get("timeout_per_seg", _DEFAULT_TIMEOUT_PER_SEG))

        # ── 分类 ───────────────────────────────────────────────────
        results: list[AudioSegmentResult | None] = [None] * len(instructions)
        to_synth: list[tuple[int, TTSInstruction, Path]] = []
        errors: list[str] = []

        for idx in range(len(instructions)):
            inst = instructions[idx]

            if idx >= n:
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False, error="skipped: beyond --limit",
                )
                continue

            voice_ref = (inst.voice_ref or "").strip() or (speaker_to_voice.get(inst.speaker) or "").strip()
            if not voice_ref:
                msg = (
                    f"missing voice_ref for speaker={inst.speaker!r}; "
                    "check voicebank_result or tts_instruction_builder fallback chain"
                )
                errors.append(f"{inst.segment_id}: {msg}")
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False, error=msg,
                )
                continue

            output_wav = audio_dir / inst.output_filename

            if output_wav.exists() and output_wav.stat().st_size > 0:
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=str(output_wav), success=True, error="",
                )
                continue

            voice_ref_path = Path(voice_ref)
            if not voice_ref_path.exists() or voice_ref_path.stat().st_size == 0:
                msg = f"voice_ref wav missing/empty: {voice_ref}"
                errors.append(f"{inst.segment_id}: {msg}")
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False, error=msg,
                )
                continue

            if dry_run:
                log_path = audio_dir / f"{Path(inst.output_filename).stem}.log"
                self._write_dry_run_log(log_path, inst, voice_ref)
                results[idx] = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False, error="dry_run: not invoked",
                )
                continue

            to_synth.append((idx, inst, voice_ref_path))

        # ── 并发合成 ────────────────────────────────────────────────
        if to_synth:
            workers = max_workers or int(self.extra_args.get("max_workers", _DEFAULT_MAX_WORKERS))
            workers = max(1, min(workers, len(to_synth)))
            print(
                f"[cosyvoice_http] synthesizing {len(to_synth)} segments "
                f"with {workers} workers (mode={self.mode}, timeout={timeout}s/seg)",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cosyvoice") as ex:
                future_to_idx = {
                    ex.submit(
                        self._synthesize_one,
                        inst, voice_ref_path, audio_dir / inst.output_filename, audio_dir, timeout,
                    ): idx
                    for idx, inst, voice_ref_path in to_synth
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as err:  # noqa: BLE001
                        inst = instructions[idx]
                        msg = f"worker exception: {type(err).__name__}: {err}"
                        errors.append(f"{inst.segment_id}: {msg}")
                        results[idx] = AudioSegmentResult(
                            segment_id=inst.segment_id, speaker=inst.speaker,
                            audio_path=None, success=False, error=msg,
                        )

        # ── 收尾 ────────────────────────────────────────────────────
        final_results: list[AudioSegmentResult] = []
        for i, r in enumerate(results):
            if r is None:
                inst = instructions[i]
                r = AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False, error="internal: result not set",
                )
                errors.append(f"{inst.segment_id}: internal result None")
            elif not r.success and not r.error.startswith("skipped"):
                errors.append(f"{r.segment_id}: {r.error}")
            final_results.append(r)

        if errors:
            (audio_dir / "errors.log").write_text(
                "\n".join(errors) + "\n", encoding="utf-8",
            )

        return final_results

    # ── 线程任务（单段合成） ─────────────────────────────────────────

    def _synthesize_one(
        self,
        inst: TTSInstruction,
        voice_ref_path: Path,
        output_wav: Path,
        audio_dir: Path,
        timeout: int,
    ) -> AudioSegmentResult:
        log_path = audio_dir / f"{Path(inst.output_filename).stem}.log"
        try:
            voice_b64 = base64.b64encode(voice_ref_path.read_bytes()).decode("ascii")
            prompt_text = self._build_prompt_text(inst)
            payload: dict[str, Any] = {
                "text": inst.text,
                "prompt_text": prompt_text,
                "prompt_audio": voice_b64,
                "mode": self.mode,
                "stream": False,
            }
            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                logf.write("=== INVOCATION ===\n")
                logf.write(f"POST {self.base_url}/v1/cosyvoice/generate\n")
                logf.write(f"text={inst.text!r}\n")
                logf.write(f"prompt_text={prompt_text!r}\n")
                logf.write(f"prompt_audio=<{len(voice_b64)} chars base64>\n")
                logf.write(f"mode={self.mode}, stream=False\n")
                logf.write(f"timeout={timeout}s, bypass_proxy={self.bypass_proxy}\n")
                logf.write("=== STYLE snapshot (already merged into prompt_text) ===\n")
                logf.write(self._format_style_snapshot(inst) + "\n")
                logf.write("=== OUTPUT ===\n")
                logf.flush()
                wav_bytes = self._post_generate(payload, timeout=timeout, logf=logf)

            if not wav_bytes or len(wav_bytes) < 44:
                return AudioSegmentResult(
                    segment_id=inst.segment_id, speaker=inst.speaker,
                    audio_path=None, success=False,
                    error=f"server returned empty/invalid wav ({len(wav_bytes)} bytes); see {log_path.name}",
                )

            # CosyVoice3 服务器返回 IEEE float wav（format=3），audio_merger 用
            # stdlib wave 模块只支持 PCM format=1，统一转成 PCM_16 再落盘。
            self._save_wav_pcm16(wav_bytes, output_wav)
            return AudioSegmentResult(
                segment_id=inst.segment_id, speaker=inst.speaker,
                audio_path=str(output_wav), success=True, error="",
            )
        except Exception as err:  # noqa: BLE001
            return AudioSegmentResult(
                segment_id=inst.segment_id, speaker=inst.speaker,
                audio_path=None, success=False,
                error=f"{type(err).__name__}: {err}",
            )

    def _save_wav_pcm16(self, wav_bytes: bytes, output_path: Path) -> None:
        """把 HTTP 返回的 wav 字节流统一保存为 PCM int16 格式。

        CosyVoice3 默认返回 IEEE float（WAVE_FORMAT_IEEE_FLOAT, format=3）；
        audio_merger 用 stdlib wave 模块只支持 PCM format=1，读 float wav
        会抛 'unknown format: 3'。无论原格式是 float 还是 PCM，转换后都得到
        统一的 PCM_16 wav（int16 → float → int16 量化误差近似为 0）。
        """
        audio, sr = sf.read(io.BytesIO(wav_bytes), format='WAV', always_2d=False)
        audio = np.clip(audio, -1.0, 1.0)
        int16_audio = (audio * 32767).astype(np.int16)
        sf.write(str(output_path), int16_audio, sr, format='WAV', subtype='PCM_16')

    # ── 内部工具 ─────────────────────────────────────────────────────

    def _build_prompt_text(self, inst: TTSInstruction) -> str:
        """根据通用字段拼 CosyVoice instruct 模式的 prompt_text。

        instruct 模式要求格式：``指令.<|endofprompt|>``

        优先级：
            1. delivery_instruction（导演层给的具体朗读指导，最准）；
            2. emotion + tone + volume + pace（兜底拼接）；
            3. 啥都没有 → "用自然平和的语气说"。
        """
        parts: list[str] = []

        delivery = (inst.delivery_instruction or "").strip().rstrip("。.")
        if delivery:
            parts.append(delivery)
        else:
            bits: list[str] = []
            if inst.emotion and inst.emotion != "neutral":
                bits.append(f"用{inst.emotion}的情绪")
            if inst.tone and inst.tone != "normal":
                bits.append(f"以{inst.tone}的语气")
            if inst.volume == "soft":
                bits.append("轻声")
            elif inst.volume == "strong":
                bits.append("有力地")
            if inst.pace < 0.9:
                bits.append("语速稍慢")
            elif inst.pace > 1.1:
                bits.append("语速稍快")
            if bits:
                parts.append("、".join(bits) + "说")
            else:
                parts.append("用自然平和的语气说")

        instruction_str = "，".join(parts)
        return f"{instruction_str}.{_ENDOFPROMPT}"

    def _format_style_snapshot(self, inst: TTSInstruction) -> str:
        stress = inst.stress_words if isinstance(inst.stress_words, list) else []
        return (
            f"emotion={inst.emotion}; intensity={inst.emotion_intensity:.2f}; "
            f"tone={inst.tone}; volume={inst.volume}; pitch={inst.pitch}; "
            f"pace={inst.pace:.2f}; pause_hint={inst.pause_hint:.2f}; "
            f"stress_words={stress}; delivery={inst.delivery_instruction!r}"
        )

    def _post_generate(
        self,
        payload: dict[str, Any],
        *,
        timeout: int,
        logf: Any,
    ) -> bytes:
        url = f"{self.base_url}/v1/cosyvoice/generate"
        proxies = {"http": None, "https": None} if self.bypass_proxy else None
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=proxies,
                verify=False,
            )
        except requests.RequestException as err:
            raise TTSError(f"HTTP 请求失败：{err}") from err

        if response.status_code >= 400:
            try:
                logf.write(f"HTTP {response.status_code} {response.reason}\n")
                logf.write(response.text[:2000] + "\n")
                logf.flush()
            except Exception:
                pass
            raise TTSError(
                f"HTTP {response.status_code} {response.reason}; "
                f"body 前 200 字符：{response.text[:200]}"
            )

        return response.content

    def _write_dry_run_log(
        self,
        log_path: Path,
        inst: TTSInstruction,
        voice_ref: str,
    ) -> None:
        prompt_text = self._build_prompt_text(inst)
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            logf.write("=== DRY RUN ===\n")
            logf.write(f"POST {self.base_url}/v1/cosyvoice/generate\n")
            logf.write(f"text={inst.text!r}\n")
            logf.write(f"prompt_text={prompt_text!r}\n")
            logf.write(f"voice_ref={voice_ref}\n")
            logf.write(f"mode={self.mode}\n")
            logf.write("=== STYLE snapshot (already merged into prompt_text) ===\n")
            logf.write(self._format_style_snapshot(inst) + "\n")
