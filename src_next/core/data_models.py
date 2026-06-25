"""src_next.core.data_models — 有声书 Agent 核心数据结构

每个 dataclass 代表数据流链路中的一个中间产物。
所有字段尽量简单，可有默认值。
"""

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 链路起点
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StoryInput:
    """
    链路起点：原始文本输入。

    Attributes
    ----------
    story_name : str
        故事名称（从文件名或用户提供提取）。
    text : str
        原始文本全文。
    source_path : str | None
        原始文件路径（可选，无文件时为 None）。
    """

    story_name: str
    text: str
    source_path: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# 链路早期产物（segment_builder 输出）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Segment:
    """
    原始文本经过切分后的最小单元。

    出现在数据流的：txt → segments

    Attributes
    ----------
    segment_id : str
        唯一编号，格式为 seg_001, seg_002 ...
    text : str
        该段的文本内容。
    speaker : str
        说话人，默认 narrator，待 analysis 层解析后更新。
    segment_type : str
        段类型，取值：
        - ``narration``：旁白叙述（包括被 quote_classifier 判定为非对白
          而并回的引号内容，如强调词、书名等）。
        - ``dialogue``：真实角色说出的对白。
        - ``inner_thought``：心理活动（心想 / 暗想等），由 quote_classifier
          从原 dialogue candidate 中细分出来。
        - ``unknown``：临时占位，上游 segment_builder 切出来但未被
          quote_classifier 处理过的 dialogue candidate 会带这个 speaker；
          segment_type 本身很少为 unknown。
    raw_index : int
        段落位置索引（从 0 开始）。core.segment_builder 会先按段落切分、
        再按引号切分，因此同一段落切出来的多个 Segment 会共享同一个 raw_index。
        analysis/story_resolver 依赖这个字段按段落重组上下文送给 LLM。
    """

    segment_id: str
    text: str
    speaker: str = "narrator"
    segment_type: str = "narration"
    raw_index: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# 链路中期产物（analysis 层输出 + voicebank 层输出）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CharacterProfile:
    """
    角色档案：从文本中提取的角色信息。

    出现在数据流的：resolved_segments → characters

    Attributes
    ----------
    name : str
        角色名称。
    role_type : str
        角色类型，narrator / character。
    gender : str | None
        性别提示（由 LLM 分析生成）。
    age_style : str | None
        年龄/声音风格提示。
    personality : str | None
        性格特点（用于朗读风格指导）。
    voice_prompt : str
        供 voicebank 层使用的音色描述提示词。
    confidence : float
        分析置信度，0.0~1.0。
    aliases : list[str]
        该角色的别名列表（不含 canonical name 本身）。由 character_analyzer
        在归并阶段填入：同一角色的不同称呼（绰号 / 亲属称谓 / 描述性称呼等）
        会合并到一个 CharacterProfile，被合并的称呼写进 aliases。
        narrator 永远为空。
    """

    name: str
    role_type: str = "character"
    gender: str | None = None
    age_style: str | None = None
    personality: str | None = None
    voice_prompt: str = ""
    confidence: float = 0.8
    aliases: list[str] = field(default_factory=list)


@dataclass
class DirectorInstruction:
    """
    导演指令：为每个 segment 提供的朗读指导。

    出现在数据流的：characters + resolved_segments → director_plan

    定位：**通用语义导演层**。字段面向"人能读懂的朗读意图"，不绑定任何具体
    TTS 后端（IndexTTS / CosyVoice / FishPro / Qwen TTS 等）。后续
    ``core/tts_instruction_builder.py`` 会把这些通用字段翻译成各 TTS adapter
    能消费的具体参数或 prompt。

    Attributes
    ----------
    segment_id : str
        对应的 segment 编号。
    speaker : str
        说话人。
    emotion : str
        情绪基调。常用值：neutral / warm / happy / excited / nostalgic /
        sad / gentle / anxious / playful / serious / moved / surprised /
        calm / joyful / longing 等。
    emotion_intensity : float
        情绪强度，0.0~1.0。0.3 以下内敛，0.5 适中，0.8 以上强烈。
    pace : float
        语速倍率，0.75~1.30。0.75 很慢，1.0 正常，1.30 快。
    tone : str
        语气描述。常用值：gentle / warm / serious / playful / calm /
        lively / normal。
    volume : str
        音量。``soft`` / ``normal`` / ``strong``。
    pitch : str
        音高。``low`` / ``medium_low`` / ``medium`` / ``medium_high`` / ``high``。
    pause_hint : float
        段后停顿秒数建议，0.2~1.0。
    stress_words : list[str]
        需要重读的关键词，最多 3 个，必须是原文中出现的词。
    delivery_instruction : str
        结合原文内容、人物、上下文的中文朗读指导。不允许使用
        "自然对白语气" / "平稳叙述" 等空泛表达。
    """

    segment_id: str
    speaker: str
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    pace: float = 1.0
    tone: str = "normal"
    volume: str = "normal"
    pitch: str = "medium"
    pause_hint: float = 0.0
    stress_words: list[str] = field(default_factory=list)
    delivery_instruction: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 链路中后段产物（tts 层消费前）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TTSInstruction:
    """
    TTS 合成指令：包含合成单段音频所需的全部信息。

    出现在数据流的：segments + characters + director_plan + voicebank → tts_instructions

    定位：**模型无关的通用合成指令**。字段面向"任何 TTS 后端都能理解的
    语义维度"，不绑定 IndexTTS / CosyVoice / FishPro / Qwen TTS 等具体后端。
    后续 ``src_next/tts/`` 各 adapter 负责把这些通用字段翻译成具体模型参数
    或 HTTP 请求体。

    不要在本 dataclass 加入 ``indextts_speed`` / ``cosyvoice_prompt`` /
    ``fishpro_temperature`` / ``qwen_voice_id`` 这类模型专用字段。模型专用
    参数应该在 adapter 内部根据通用字段推断，避免 core 层耦合到某个后端。

    Attributes
    ----------
    segment_id : str
        对应的 segment 编号。
    speaker : str
        说话人。
    text : str
        要合成的文本。
    segment_type : str
        段落类型（``narration`` / ``dialogue`` / ``inner_thought``）。
        从 Segment.segment_type 拷贝；TTS adapter 可以根据它走不同分支。
    voice_ref : str
        音色参考文件路径（由 voicebank 层填充）。空字符串表示没拿到
        对应 speaker 的 voice reference（已经在 metadata 里标记）。
    emotion : str
        情绪基调（从 DirectorInstruction 复制）。
    emotion_intensity : float
        情绪强度 0.0~1.0。
    pace : float
        语速倍率 0.75~1.30。
    tone : str
        语气描述（gentle / warm / serious / playful / calm / lively / normal 等）。
    volume : str
        音量（``soft`` / ``normal`` / ``strong``）。
    pitch : str
        音高（``low`` / ``medium_low`` / ``medium`` / ``medium_high`` / ``high``）。
    pause_hint : float
        段后停顿秒数 0.2~1.0。
    stress_words : list[str]
        需要重读的关键词，最多 3 个。
    delivery_instruction : str
        综合朗读指导（结合原文内容的中文描述）。
    output_filename : str
        建议的音频输出文件名，格式 ``seg_001.wav``。
    metadata : dict[str, Any]
        调试 / 审计字段。常驻键：
        * ``source_segment_type``：原始 segment_type（和 segment_type 字段重复，
          留作 fallback 痕迹）。
        * ``has_director_instruction``：本段是否拿到 LLM 产出的 director。
        * ``has_voice_ref``：是否拿到 speaker 对应的 voice reference。
        * ``missing_voice_ref``（仅当 has_voice_ref=False）：True。
        * ``voice_ref_fallback``（仅当 fallback 到 narrator / 空）：标注走哪个兜底。
    """

    segment_id: str
    speaker: str
    text: str
    segment_type: str = "narration"
    voice_ref: str = ""
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    pace: float = 1.0
    tone: str = "normal"
    volume: str = "normal"
    pitch: str = "medium"
    pause_hint: float = 0.4
    stress_words: list[str] = field(default_factory=list)
    delivery_instruction: str = ""
    output_filename: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# voicebank 层产物
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VoicebankResult:
    """
    voicebank 层产出：每个 speaker 对应的音色参考。

    出现在数据流的：characters → voicebank_result

    Attributes
    ----------
    speaker_to_voice : dict[str, str]
        speaker 名称到音色文件路径的映射。
        示例：{"narrator": "voicebank/narrator.wav", "小明": "voicebank/小明.wav"}
    voicebank_dir : str | None
        voicebank 输出目录路径。
    backend : str
        使用的 voicebank backend 名称（如 mock / cosyvoice / indextts）。
    success : bool
        是否全部成功生成。
    """

    speaker_to_voice: dict[str, str] = field(default_factory=dict)
    voicebank_dir: str | None = None
    backend: str = "mock"
    success: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# audio 层产物（tts 层 + audio_merger 层输出）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AudioSegmentResult:
    """
    单个 segment 的音频合成结果。

    出现在数据流的：tts_adapter → audio_segments（内部列表元素）

    Attributes
    ----------
    segment_id : str
        对应的 segment 编号。
    speaker : str
        说话人。
    audio_path : str | None
        生成的音频文件路径（失败时为 None）。
    success : bool
        是否成功合成。
    error : str
        错误信息（成功时为空字符串）。
    """

    segment_id: str
    speaker: str
    audio_path: str | None = None
    success: bool = True
    error: str = ""


@dataclass
class AudioResult:
    """
    音频合并结果：所有 segment 合并后的最终音频。

    出现在数据流的：audio_segments → audio_result → pipeline_result

    Attributes
    ----------
    final_audio : str | None
        合并后的最终音频文件路径。
    audio_segments : list[AudioSegmentResult]
        所有 segment 的单独音频结果。
    duration_seconds : float
        音频总时长（秒）。
    success : bool
        是否成功合并。
    """

    final_audio: str | None = None
    audio_segments: list[AudioSegmentResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# 链路终点
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    """
    完整 pipeline 运行结果：链路终点。

    出现在数据流的：最后汇总

    Attributes
    ----------
    story_name : str
        故事名称。
    output_dir : str
        输出根目录。
    final_audio : str | None
        最终音频文件路径。
    success : bool
        是否完全成功。
    stage_timings : dict[str, float]
        各阶段耗时（秒），键为阶段名称。
    artifacts : dict[str, str]
        中间产物路径映射，如 {"segments": "json/segments.json", ...}
    error : str
        错误信息（成功时为空字符串）。
    pipeline_summary : dict[str, Any]
        真实 pipeline 的完整耗时 / RTF / 阶段明细。mock pipeline 留空 dict。
        真实 pipeline 填入 total_time_sec / analysis_time_sec / voicebank_time_sec /
        tts_time_sec / merge_time_sec / final_audio_duration_sec / rtf /
        output_dir / final_audio_path / stages[]（每项含 stage/status/elapsed_sec/
        mode/output）。详见 src_next/core/audiobook_pipeline.py。
    """

    story_name: str
    output_dir: str
    final_audio: str | None = None
    success: bool = True
    stage_timings: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""
    pipeline_summary: dict[str, Any] = field(default_factory=dict)
