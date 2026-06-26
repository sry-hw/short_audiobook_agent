"""src_next/tts/s2pro_adapter.py

Fish Audio S2-Pro 4B TTS adapter（控制信号增强层 + 音色克隆路由）。

===========================================================================
定位：**控制信号增强层**（control signal enhancement layer）
===========================================================================

不是简单 TTS wrapper。本 adapter 在通用 TTSInstruction 和 S2Pro API 之间
做两层翻译：

1. **风格控制翻译**：把通用字段（emotion / pace / volume / pitch / pause_hint
   / stress_words / delivery_instruction）翻译成 S2Pro 的内联标签 + 全局
   instruction。
   - emotion=sad + intensity=0.7 → ``[sigh][sad][speak slowly][quiet]...``
   - stress_words=["松果"] → 在原文中包裹 ``[emphasis]松果``
   - pause_hint=0.9 → 段末追加 ``[pause]``

2. **音色路由翻译**：从 voicebank_result 查 speaker 对应的 wav 路径，
   写入 reference_audio 字段。S2Pro 通过 reference_audio + prompt_text +
   enable_reference_audio=true 实现音色克隆。

===========================================================================
当前阶段（v1）：转换验证，不调真实 S2Pro API
===========================================================================

v1 只产出 ``S2ProRenderResult``（包含 s2pro_text / instruction /
reference_audio_path / params / debug_tags），把每段的转换结果落盘到
``<audio_dir>/<segment_id>.s2pro.txt``，便于人工 / 脚本核验。

v2 会改为真实 HTTP 调用（需要先扩展 ``/v1/voicegen/generate`` wrapper
接受 reference_audio）。

===========================================================================
数据流位置
===========================================================================

    TTSInstruction[] + VoicebankResult
        → S2ProTTSAdapter.convert_instructions(...) 或 .synthesize(...)
        → S2ProRenderResult[]（每段一个，与 instructions 1:1 对应）

S2ProRenderResult 直接对应 S2Pro API 的 multipart/form-data 字段：

    result.s2pro_text           → API ``text`` 字段（含内联标签）
    result.instruction          → API ``instruction`` 字段（全局风格）
    result.reference_audio_path → API ``reference_audio`` 文件
    result.prompt_text          → API ``prompt_text`` 字段（必须与 wav 匹配）
    result.enable_reference_audio → API ``enable_reference_audio`` 标志
    result.params               → API temperature / top_p / max_new_tokens

参考文档：
    * ``usage_guide_s2pro.md``（项目根，含本地 wrapper 和云 API 对比）
    * Fish Audio 官方：https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech
    * fish-speech 源码：https://github.com/fishaudio/fish-speech
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src_next.core.data_models import AudioSegmentResult, TTSInstruction, VoicebankResult

from .base import BaseTTSAdapter, TTSError


# ─── 默认值 ──────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "http://10.50.121.102:8006"
_DEFAULT_OUTPUT_SUBDIR = "audio_segments"
_DEFAULT_MODEL = "s2pro-4b"
_DEFAULT_DENSITY = "medium"  # low / medium / high
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_P = 0.7
_DEFAULT_MAX_NEW_TOKENS = 4096
_DEFAULT_TIMEOUT_PER_SEG = 300

_DENSITY_LEVELS = ("low", "medium", "high")


# ─── 转换规则表（module-level constants） ────────────────────────────────────

# DirectorInstruction.emotion → S2Pro 内联标签
# 优先用 S2Pro 文档明确的固定标签；无对应的用自由文本描述（也合法）
_EMOTION_TO_S2PRO_TAG: dict[str, str] = {
    # S2Pro 固定标签（docs 明确列出）
    "excited": "[excited]",
    "sad": "[sad]",
    "angry": "[angry]",
    "surprised": "[surprised]",
    "fearful": "[fearful]",
    # 自由文本描述（S2Pro 支持任意自然语言）
    "happy": "[happy]",
    "joyful": "[happy]",
    "calm": "[calm]",
    "nostalgic": "[nostalgic]",
    "longing": "[nostalgic]",
    "moved": "[moved]",
    "warm": "[warm tone]",
    "gentle": "[gentle voice]",
    "playful": "[playful tone]",
    "serious": "[serious tone]",
    "anxious": "[anxious]",
    # neutral / unknown → 留空（omit）
}

# emotion_intensity 高位时推断的声音效果标签（受 max_tag_density 控制）
# (emotion, intensity_threshold, tag) → prepend 到 s2pro_text 头部
_INTENSITY_INFER_PRE: list[tuple[str, float, str]] = [
    ("sad",      0.70, "[sigh]"),
    ("anxious",  0.70, "[inhale]"),
    ("moved",    0.70, "[exhale]"),
    ("angry",    0.85, "[shouting]"),
]

# (emotion, intensity_threshold, tag) → append 到 s2pro_text 尾部（在 pause_tag 之前）
_INTENSITY_INFER_POST: list[tuple[str, float, str]] = [
    ("joyful",   0.85, "[laughing]"),
    ("excited",  0.85, "[laughing]"),
]


# ─── S2ProRenderResult 数据结构 ──────────────────────────────────────────────


@dataclass
class S2ProRenderResult:
    """S2Pro adapter 转换单条 TTSInstruction 的产物。

    直接对应 S2Pro API 的 multipart/form-data 字段 + 调试信息。
    字段映射见模块 docstring。

    Attributes
    ----------
    segment_id : str
        段落 ID（与原 TTSInstruction.segment_id 一致）。
    speaker : str
        说话人名（多说话人拼接时映射到 ``<|speaker:N|>`` 的 N；v1 不拼接）。
    original_text : str
        原文（未动）；调试 / 对照用。
    s2pro_text : str
        加了内联标签的文本（→ API ``text`` 字段）。
    instruction : str
        全局风格指令（→ API ``instruction`` 字段）。
    reference_audio_path : str
        该 speaker 的 voicebank wav 路径（→ API ``reference_audio`` 文件字段）。
        空字符串表示未拿到 voice_ref（v1 测试模式或 voicebank 失败）。
    enable_reference_audio : bool
        是否启用音色克隆。False 则用模型默认音色（每次调用随机）。
    prompt_text : str
        reference_audio 的转录文本（→ API ``prompt_text`` 字段）。
        **必须与 wav 内容匹配**，否则克隆效果劣化。
    params : dict
        透传给 API 的生成参数：temperature / top_p / max_new_tokens。
    debug_tags : dict
        转换诊断信息：哪些标签被加、推断来源、跳过原因、voice_ref 来源。
    """

    segment_id: str
    speaker: str
    original_text: str
    s2pro_text: str
    instruction: str
    reference_audio_path: str
    enable_reference_audio: bool
    prompt_text: str
    params: dict[str, Any] = field(default_factory=dict)
    debug_tags: dict[str, Any] = field(default_factory=dict)


# ─── S2ProTTSAdapter ────────────────────────────────────────────────────────


class S2ProTTSAdapter(BaseTTSAdapter):
    """Fish Audio S2-Pro 4B adapter（控制信号增强层 + 音色克隆路由）。

    v1：只做 instruction → S2Pro 参数转换；不调真实 HTTP。
    v2：会加真实合成（参考 cosyvoice_http.py 模式）。
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        output_subdir: str = _DEFAULT_OUTPUT_SUBDIR,
        extra_args: dict[str, Any] | None = None,
        **_unused: Any,
    ) -> None:
        self.base_url = (base_url or _DEFAULT_BASE_URL).strip().rstrip("/")
        self.output_subdir = (
            (output_subdir or _DEFAULT_OUTPUT_SUBDIR).strip() or _DEFAULT_OUTPUT_SUBDIR
        )

        # 解析 extra_args
        extra_args = dict(extra_args or {})
        self.model: str = str(extra_args.get("model", _DEFAULT_MODEL))
        self.enable_inline_tags: bool = bool(extra_args.get("enable_inline_tags", True))
        self.max_tag_density: str = str(extra_args.get("max_tag_density", _DEFAULT_DENSITY))
        if self.max_tag_density not in _DENSITY_LEVELS:
            self.max_tag_density = _DEFAULT_DENSITY

        # 音色克隆配置
        self.enable_reference_audio: bool = bool(extra_args.get("enable_reference_audio", True))
        # prompt_text 必须与 voicebank wav 内容匹配；profile 应与 voicebank.reference_text 一致
        self.prompt_text: str = str(extra_args.get("prompt_text", ""))

        # 生成参数
        self.temperature: float = float(extra_args.get("temperature", _DEFAULT_TEMPERATURE))
        self.top_p: float = float(extra_args.get("top_p", _DEFAULT_TOP_P))
        self.max_new_tokens: int = int(extra_args.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))

        # 工程参数
        self.timeout_per_seg: int = int(extra_args.get("timeout_per_seg", _DEFAULT_TIMEOUT_PER_SEG))
        self.bypass_proxy: bool = bool(extra_args.get("bypass_proxy", True))

    # ─── 公开 API ────────────────────────────────────────────────────────

    def convert_instructions(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult | None = None,
    ) -> list[S2ProRenderResult]:
        """批量转换 TTSInstruction 列表为 S2ProRenderResult 列表。

        Args:
            instructions: 通用 TTS 指令列表。
            voicebank_result: 可选；提供时会从中查 speaker → wav 路径，
                写入 result.reference_audio_path。None 时该字段为空字符串。

        Returns:
            S2ProRenderResult 列表，长度 = len(instructions)，顺序一致。
        """
        return [
            self.convert_instruction(inst, voicebank_result)
            for inst in instructions
        ]

    def convert_instruction(
        self,
        instruction: TTSInstruction,
        voicebank_result: VoicebankResult | None = None,
    ) -> S2ProRenderResult:
        """转换单条 TTSInstruction 为 S2ProRenderResult。

        本方法是 adapter 的核心。它调 :meth:`_convert_instruction_to_s2pro_text`
        做标签转换，并补充 reference_audio 路由信息。
        """
        # 1. 标签转换 + instruction 全局风格
        s2pro_text, instruction_text, debug_tags = self._convert_instruction_to_s2pro_text(
            instruction
        )

        # 2. reference_audio 路由
        ref_path, voice_ref_status = self._resolve_reference_audio(
            instruction, voicebank_result
        )
        debug_tags["voice_ref_status"] = voice_ref_status

        # 3. params 透传
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

        return S2ProRenderResult(
            segment_id=instruction.segment_id,
            speaker=instruction.speaker,
            original_text=instruction.text,
            s2pro_text=s2pro_text,
            instruction=instruction_text,
            reference_audio_path=ref_path,
            enable_reference_audio=self.enable_reference_audio and bool(ref_path),
            prompt_text=self.prompt_text,
            params=params,
            debug_tags=debug_tags,
        )

    # ─── BaseTTSAdapter 实现（v1 = convert-only，不调 HTTP） ─────────────

    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        *,
        dry_run: bool = False,
        limit: int = 0,
        **_kwargs: Any,
    ) -> list[AudioSegmentResult]:
        """v1：转换 + 把 s2pro_text 落盘到 ``<audio_dir>/<seg>.s2pro.txt``。

        返回 AudioSegmentResult 列表；success=False 标记"未实际合成"
        （error 字段说明原因）。v2 会改为真实 HTTP 调用。

        Args:
            limit: > 0 时只处理前 N 条（调试用）。
        """
        audio_dir = Path(output_dir).expanduser() / self.output_subdir
        audio_dir.mkdir(parents=True, exist_ok=True)

        if limit > 0:
            instructions = instructions[:limit]

        results = self.convert_instructions(instructions, voicebank_result)

        # 落盘转换结果（json 全量 + s2pro_text 纯文本）
        out_records: list[dict[str, Any]] = []
        audio_segment_results: list[AudioSegmentResult] = []

        for r in results:
            # 纯文本（人工核验用）
            txt_path = audio_dir / f"{r.segment_id}.s2pro.txt"
            txt_path.write_text(
                f"# instruction: {r.instruction}\n"
                f"# reference_audio: {r.reference_audio_path}\n"
                f"# enable_clone: {r.enable_reference_audio}\n"
                f"# prompt_text: {r.prompt_text}\n"
                f"# params: {json.dumps(r.params, ensure_ascii=False)}\n\n"
                f"{r.s2pro_text}\n",
                encoding="utf-8",
            )

            # 结构化 json（自动化核验用）
            out_records.append(asdict(r))

            # AudioSegmentResult 占位（success=False：v1 未实际合成）
            audio_segment_results.append(
                AudioSegmentResult(
                    segment_id=r.segment_id,
                    speaker=r.speaker,
                    audio_path=None,
                    success=False,
                    error=(
                        "S2Pro v1: convert-only mode, no HTTP call. "
                        f"s2pro_text saved to {txt_path.name}"
                    ),
                )
            )

        # 汇总 json
        summary_path = audio_dir / "s2pro_render_results.json"
        summary_path.write_text(
            json.dumps(out_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return audio_segment_results

    # ─── 内部：reference_audio 路由 ──────────────────────────────────────

    def _resolve_reference_audio(
        self,
        instruction: TTSInstruction,
        voicebank_result: VoicebankResult | None,
    ) -> tuple[str, str]:
        """从 voicebank_result 查 speaker 对应 wav 路径。

        优先级：
            1. voicebank_result.speaker_to_voice[speaker]
            2. instruction.voice_ref（tts_instruction_builder 已填）
            3. voicebank_result.speaker_to_voice["narrator"]（fallback）
            4. 空字符串（reference_audio 不可用）

        Returns:
            (wav_path, status) — status ∈ {"ok", "instruction_voice_ref",
            "fallback_to_narrator", "missing"}
        """
        speaker = instruction.speaker or "narrator"

        # 1. voicebank_result 优先
        if voicebank_result and voicebank_result.speaker_to_voice:
            wav = voicebank_result.speaker_to_voice.get(speaker)
            if wav:
                return str(wav), "ok"
            # 3. narrator fallback
            if speaker != "narrator":
                wav = voicebank_result.speaker_to_voice.get("narrator")
                if wav:
                    return str(wav), "fallback_to_narrator"

        # 2. instruction.voice_ref
        if instruction.voice_ref:
            return str(instruction.voice_ref), "instruction_voice_ref"

        # 4. 都没有
        return "", "missing"

    # ─── 内部：核心转换逻辑 ──────────────────────────────────────────────

    def _convert_instruction_to_s2pro_text(
        self,
        instruction: TTSInstruction,
    ) -> tuple[str, str, dict[str, Any]]:
        """把 TTSInstruction 转换为 (s2pro_text, instruction, debug_tags)。

        s2pro_text 构造顺序（见 plan 第 4.10 节）::

            [inferred_pre_tags] + [emotion_tag] + [pace_tag] + [volume_tag]
            + [pitch_tag]
            + 标注后的原文（stress_words 已插 [emphasis]）
            + [inferred_post_tags] + [pause_tag]
        """
        debug: dict[str, Any] = {
            "skipped": [],
            "density": self.max_tag_density,
            "inline_tags_enabled": self.enable_inline_tags,
        }

        if not self.enable_inline_tags:
            # 关闭内联标签：原文直出，instruction 仍用 delivery_instruction
            s2pro_text = instruction.text
            instruction_text = self._build_instruction(instruction, debug)
            return s2pro_text, instruction_text, debug

        # ── step 1: emotion tag ──
        emotion_tag = self._emotion_to_tag(instruction.emotion, debug)

        # ── step 2: pace tag ──
        pace_tag = self._pace_to_tag(instruction.pace, debug)

        # ── step 3: volume tag ──
        volume_tag = self._volume_to_tag(instruction.volume, debug)

        # ── step 4: pitch tag ──
        pitch_tag = self._pitch_to_tag(instruction.pitch, debug)

        # ── step 5: stress words → 内联 [emphasis] 包裹 ──
        stress_wraps, annotated_text = self._apply_stress_words(
            instruction.text, instruction.stress_words, debug
        )

        # ── step 6: intensity-driven inferred tags ──
        inferred_pre, inferred_post = self._infer_intensity_tags(
            instruction.emotion, instruction.emotion_intensity, debug
        )

        # ── step 7: pause tag（段末）──
        pause_tag = self._pause_hint_to_tag(instruction.pause_hint, debug)

        # 按 density 过滤标签
        pre_tags = self._filter_by_density(
            inferred_pre, [emotion_tag, pace_tag, volume_tag, pitch_tag]
        )
        post_tags = self._filter_by_density(
            inferred_post, [pause_tag]
        )

        # 拼最终 s2pro_text
        prefix = "".join(pre_tags)
        suffix = "".join(post_tags)
        # 前缀后加一个空格让原文不被标签粘连
        if prefix and not prefix.endswith(" "):
            prefix = prefix + " "
        s2pro_text = f"{prefix}{annotated_text}{suffix}"

        # 全局 instruction
        instruction_text = self._build_instruction(instruction, debug)

        # 写入 debug_tags 摘要
        debug.update({
            "emotion_tag": emotion_tag,
            "pace_tag": pace_tag,
            "volume_tag": volume_tag,
            "pitch_tag": pitch_tag,
            "pause_tag": pause_tag,
            "stress_wraps": stress_wraps,
            "inferred_pre_tags": inferred_pre,
            "inferred_post_tags": inferred_post,
            "final_pre_tags": pre_tags,
            "final_post_tags": post_tags,
        })

        return s2pro_text, instruction_text, debug

    # ─── 单字段映射函数 ──────────────────────────────────────────────────

    def _emotion_to_tag(self, emotion: str, debug: dict[str, Any]) -> str:
        emotion = (emotion or "").strip().lower()
        if not emotion or emotion == "neutral":
            debug["skipped"].append("emotion=neutral → omit")
            return ""
        tag = _EMOTION_TO_S2PRO_TAG.get(emotion)
        if tag is None:
            debug["skipped"].append(f"emotion={emotion!r} unknown → omit")
            return ""
        return tag

    def _pace_to_tag(self, pace: float, debug: dict[str, Any]) -> str:
        try:
            p = float(pace)
        except (TypeError, ValueError):
            debug["skipped"].append(f"pace={pace!r} not numeric → omit")
            return ""
        if p <= 0.90:
            return "[speak slowly]"
        if p >= 1.10:
            return "[speak quickly]"
        debug["skipped"].append(f"pace={p:.2f} normal → omit")
        return ""

    def _volume_to_tag(self, volume: str, debug: dict[str, Any]) -> str:
        v = (volume or "").strip().lower()
        if v == "soft":
            return "[quiet]"
        if v == "strong":
            return "[loud]"
        if v == "normal":
            debug["skipped"].append("volume=normal → omit")
            return ""
        debug["skipped"].append(f"volume={volume!r} unknown → omit")
        return ""

    def _pitch_to_tag(self, pitch: str, debug: dict[str, Any]) -> str:
        p = (pitch or "").strip().lower()
        if p in ("low", "medium_low"):
            return "[pitch down]"
        if p in ("medium_high", "high"):
            return "[pitch up]"
        if p == "medium":
            debug["skipped"].append("pitch=medium → omit")
            return ""
        debug["skipped"].append(f"pitch={pitch!r} unknown → omit")
        return ""

    def _pause_hint_to_tag(self, pause_hint: float, debug: dict[str, Any]) -> str:
        try:
            ph = float(pause_hint)
        except (TypeError, ValueError):
            debug["skipped"].append(f"pause_hint={pause_hint!r} not numeric → omit")
            return ""
        if ph >= 0.8:
            return "[pause]"
        if ph >= 0.4:
            return "[short pause]"
        debug["skipped"].append(f"pause_hint={ph:.2f} < 0.4 → omit")
        return ""

    def _apply_stress_words(
        self,
        text: str,
        stress_words: list[str],
        debug: dict[str, Any],
    ) -> tuple[list[str], str]:
        """在原文中首次出现位置前插入 [emphasis]。

        Returns:
            (stress_wraps, annotated_text)
            stress_wraps: 命中的项，形如 ``["[emphasis]松果"]``
            annotated_text: 标注后的原文（未命中的 stress_word 不动）
        """
        if not stress_words or not text:
            debug["stress_wraps_missed"] = list(stress_words or [])
            return [], text

        stress_wraps: list[str] = []
        annotated = text
        # 从后往前替换，避免索引偏移
        # 但每个 word 只替换首次出现，所以从前往后查到 first index
        for word in stress_words:
            if not isinstance(word, str) or not word:
                continue
            idx = annotated.find(word)
            if idx < 0:
                debug["skipped"].append(f"stress_word={word!r} not in text → skip")
                continue
            # 替换 annotated 中 idx 位置的 word → [emphasis]word
            annotated = annotated[:idx] + "[emphasis]" + annotated[idx:]
            stress_wraps.append(f"[emphasis]{word}")
            # 后续查找要跳过刚插入的 [emphasis]
            # 因为我们按 word 顺序处理，且只标记首次，这里不再复杂处理
        return stress_wraps, annotated

    def _infer_intensity_tags(
        self,
        emotion: str,
        intensity: float,
        debug: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """根据 emotion + intensity 推断附加声音效果标签。

        受 max_tag_density 控制：只有 high 才启用。
        """
        if self.max_tag_density != "high":
            debug["skipped"].append(
                f"inferred_tags disabled (density={self.max_tag_density}, need=high)"
            )
            return [], []

        emotion = (emotion or "").strip().lower()
        try:
            inten = float(intensity)
        except (TypeError, ValueError):
            return [], []

        pre: list[str] = []
        post: list[str] = []
        for emo, threshold, tag in _INTENSITY_INFER_PRE:
            if emotion == emo and inten >= threshold:
                pre.append(tag)
        for emo, threshold, tag in _INTENSITY_INFER_POST:
            if emotion == emo and inten >= threshold:
                post.append(tag)
        return pre, post

    # ─── 标签密度过滤 ────────────────────────────────────────────────────

    def _filter_by_density(
        self,
        inferred_tags: list[str],
        rule_tags: list[str],
    ) -> list[str]:
        """按 max_tag_density 合并标签。

        顺序：inferred_tags 在前，rule_tags 在后（both for pre 和 post）。
        例如 pre: ``[sigh][sad][speak slowly][quiet][pitch down]``
              post: ``[laughing][pause]``

        - low: 只保留 rule_tags 中的 emotion + pause（其余 omit；inferred 全 omit）
        - medium: rule_tags 全保留；inferred_tags 全 omit
        - high: inferred_tags + rule_tags 全保留
        """
        if self.max_tag_density == "low":
            # 只留 emotion 和 pause（pace / volume / pitch / inferred 全 omit）
            filtered_rules = [t for t in rule_tags if t.startswith((
                "[excited", "[sad", "[angry", "[surprised", "[happy",
                "[calm", "[nostalgic", "[moved", "[warm", "[gentle",
                "[playful", "[serious", "[anxious", "[fearful",
                "[pause", "[short pause",
            ))]
            return filtered_rules
        if self.max_tag_density == "medium":
            return list(rule_tags)
        # high: inferred 在前，rule 在后
        return list(inferred_tags) + list(rule_tags)

    # ─── 全局 instruction 构造 ───────────────────────────────────────────

    def _build_instruction(
        self,
        instruction: TTSInstruction,
        debug: dict[str, Any],
    ) -> str:
        """构造 S2Pro API 的全局 instruction 字段。

        优先用 delivery_instruction（导演层给的具体朗读指导）。
        空时按 emotion + pace 拼一个最小 fallback。
        """
        di = (instruction.delivery_instruction or "").strip()
        if di:
            debug["instruction_source"] = "delivery_instruction"
            return di

        # fallback
        emotion_desc = instruction.emotion or "neutral"
        if instruction.pace and float(instruction.pace or 1.0) <= 0.90:
            pace_desc = "语速偏慢"
        elif instruction.pace and float(instruction.pace or 1.0) >= 1.10:
            pace_desc = "语速偏快"
        else:
            pace_desc = "语速自然"
        fallback = f"以{emotion_desc}的语气朗读，{pace_desc}"
        debug["instruction_source"] = "fallback_emotion_pace"
        return fallback
