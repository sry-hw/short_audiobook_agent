"""src_next/analysis/story_director.py

导演计划生成（每段一个 DirectorInstruction）。

数据流位置：
    Segment[] + CharacterProfile[]
        → generate_director_plan(segments, characters, llm_client)
        → DirectorInstruction[]（和 segments 一一对应，顺序一致）

输出定位：
    **通用语义导演层**。每个字段都是"人能读懂的朗读意图"，不绑定任何具体
    TTS 后端（IndexTTS / CosyVoice / FishPro / Qwen TTS 等）。后续
    ``core/tts_instruction_builder.py`` 会把这些字段翻译成各 TTS adapter
    的具体参数或 prompt。

字段集合（11 个）：
    segment_id, speaker, emotion, emotion_intensity, pace, tone,
    volume, pitch, pause_hint, stress_words, delivery_instruction

策略：
    1. 一次性把所有 segment + 角色档案打包丢给 LLM。
    2. LLM 返回 ``segment_directions`` 列表，每条带 11 个字段。
    3. 严格清洗每个字段（合法值 / 范围 clamp / 类型转换）。
    4. LLM 没覆盖的 segment → 按 narrator / dialogue + 文本关键词 +
       speaker 年龄段给细粒度 fallback（不再用 "自然对白语气" / "平稳叙述"）。

参考旧 src/story_director.py：
- segment_directions 一一对应；fallback 兜底。
- 比旧 src 多了 emotion_intensity / volume / pitch / stress_words 四个字段。
- 去掉 overall_style / emphasis_words / needs_review（v1 不要）。
"""

from __future__ import annotations

from typing import Any

from ..core.data_models import CharacterProfile, DirectorInstruction, Segment
from ..llm.base import BaseLLMClient, LLMError


# ── 合法值集合 ──────────────────────────────────────────────────────────────

# emotion 用一个宽松的白名单：用户规格的 15 个值 + 旧版的 3 个值（angry /
# fearful / disgusted）都接受，避免 LLM 输出合法情绪时被错误 fallback。
_VALID_EMOTIONS = {
    # 用户规格中的 15 个
    "neutral", "warm", "happy", "excited", "nostalgic", "sad",
    "gentle", "anxious", "playful", "serious", "moved", "surprised",
    "calm", "joyful", "longing",
    # 兼容旧版清洗函数的 3 个
    "angry", "fearful", "disgusted",
}

_VALID_TONES = {
    "gentle", "warm", "serious", "playful", "calm", "lively", "normal",
    # 兼容旧版
    "sharp", "soft", "deep", "bright",
}

_VALID_VOLUMES = {"soft", "normal", "strong"}

_VALID_PITCHES = {"low", "medium_low", "medium", "medium_high", "high"}

# pace / pause_hint / emotion_intensity 的合法范围
_PACE_MIN, _PACE_MAX = 0.75, 1.30
_PAUSE_MIN, _PAUSE_MAX = 0.4, 1.5
_INTENSITY_MIN, _INTENSITY_MAX = 0.0, 1.0

# stress_words 最多多少个
_MAX_STRESS_WORDS = 3

# delivery_instruction 长度约束
_DELIVERY_MIN_LEN = 8
_DELIVERY_MAX_LEN = 80


# ── LLM prompt ─────────────────────────────────────────────────────────────

_DIRECTOR_SYSTEM_PROMPT = """你是一个中文有声书导演。根据每个文本片段的内容、说话人、角色档案，生成**细粒度**的朗读指导。

严格输出 JSON，结构如下：

{
  "segment_directions": [
    {
      "segment_id": "seg_001",
      "speaker": "narrator",
      "emotion": "nostalgic",
      "emotion_intensity": 0.65,
      "pace": 0.88,
      "tone": "gentle",
      "volume": "soft",
      "pitch": "medium_low",
      "pause_hint": 0.6,
      "stress_words": ["故乡", "桂花"],
      "delivery_instruction": "语气温柔而怀念，前半句平稳，后半句稍慢，突出对故乡桂花的思念。"
    }
  ]
}

字段约束：

- emotion（情绪）：neutral / warm / happy / excited / nostalgic / sad / gentle / anxious / playful / serious / moved / surprised / calm / joyful / longing 等。
- emotion_intensity（情绪强度）：0.0~1.0 浮点数。0.3 以下内敛，0.5 适中，0.8 以上强烈。
- pace（语速倍率）：0.75~1.30。0.75=很慢（深沉、回忆），0.9=稍慢（温柔叙述），1.0=正常，1.15=稍快（兴奋），1.30=快（紧张、激动）。
- tone（语气）：gentle / warm / serious / playful / calm / lively / normal。
- volume（音量）：soft / normal / strong。
- pitch（音高）：low / medium_low / medium / medium_high / high。
- pause_hint（段后停顿秒数）：0.4~1.5。**段间停顿偏长比偏短好**——中文 TTS 输出本身连贯，太短的 pause 听起来仍然紧凑。
  * 段内对白切换（dialogue → narration 或 narration → dialogue）：≥ 0.5 秒
  * 段落结尾（segment 是同段落最后一段 / 故事结尾）：≥ 0.8 秒
  * 童话 / 儿童故事：整体偏长（让小听众有消化时间）
  * 悬疑 / 紧张场景：可以适度缩短但不要低于 0.4 秒
- stress_words（重读词）：1~3 个，**必须是原文中出现的词**，不要造词。
- delivery_instruction（朗读指导）：
  * 必须是中文，10~50 字
  * **禁止使用** "自然对白语气"、"平稳叙述"、"语气自然" 等空泛表达
  * 必须结合 segment 的文本内容、说话人、情感色彩给出具体指导
  * 要说出语气、节奏、情感色彩三个维度的信息

判断依据：

- **narrator 段**（旁白）：
  * 回忆性叙述（含 "故乡/童年/想起/当年/家乡/记忆"）→ nostalgic，pace 0.85~0.95，tone warm/gentle，volume soft
  * 描写性叙述（景物、环境）→ calm，pace 0.95，tone calm，volume normal
  * 快乐场景描写（含 "快乐/高兴/喜欢/开心"）→ joyful，pace 1.05，tone lively，volume normal
  * 温馨场景描写（含 "温暖/幸福/母爱"）→ warm，pace 0.95，tone warm，volume soft
  * 悲伤场景描写（含 "悲伤/难过/失落"）→ sad，pace 0.85，tone serious，volume soft
  * 转折 / 高潮处可以适当加快 / 加强

- **dialogue 段**（对白）：
  * 童年视角的 "我" → playful / excited / curious，pace 1.05~1.15，pitch medium_high
  * 长辈（母亲 / 老人）→ gentle / warm / serious，pace 0.9，volume soft，pitch medium_low
  * 急切 / 兴奋台词（含 "！" 或连续短句）→ pace ↑、volume ↑
  * 温柔 / 思念台词（含 "想/念/记得"）→ pace ↓、volume soft
  * 疑问台词（含 "？"）→ pitch ↑

- **inner_thought 段**（心理活动）：
  * volume soft，pace 略慢，反映内心独白感

- segment_directions 数量必须等于输入 segment 数量
- 每个 segment_id 都必须出现且唯一
- 不要输出任何具体 TTS 后端的专用参数（如 indextts_speed / fishpro_temperature）
"""


# ── 入口函数 ────────────────────────────────────────────────────────────────

def generate_director_plan(
    segments: list[Segment],
    characters: list[CharacterProfile],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[DirectorInstruction]:
    """为每个 segment 生成一条 DirectorInstruction（1:1）。

    Args:
        segments: resolved segments（speaker 已识别，quote_classifier 已合并）。
        characters: ``analyze_characters`` 的输出（含 narrator）。
        llm_client: 任意 ``BaseLLMClient`` 实现。
        story_context: 故事上下文（可选）。

    Returns:
        DirectorInstruction 列表，长度严格等于 segments，按 segment 顺序排列。
    """
    if not segments:
        return []

    char_map = {c.name: c for c in characters}
    llm_dirs = _direct_via_llm(segments, characters, llm_client, story_context)
    id_to_dir = {d["segment_id"]: d for d in llm_dirs}

    plan: list[DirectorInstruction] = []
    for seg in segments:
        raw = id_to_dir.get(seg.segment_id)
        if raw:
            plan.append(_build_instruction_from_llm(seg, raw))
        else:
            plan.append(_fallback_instruction(seg, char_map.get(seg.speaker)))
    return plan


# ── LLM 调用 ────────────────────────────────────────────────────────────────

def _direct_via_llm(
    segments: list[Segment],
    characters: list[CharacterProfile],
    llm_client: BaseLLMClient,
    story_context: str,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    prompt = _build_director_prompt(segments, characters, story_context)
    try:
        # director_plan 输出特别大（每段 ~250 tokens × 段数）：
        # 1) 默认 max_tokens=1024 不够，必须显式拉高。
        #    估算：每段 ~400 tokens + 512 余量。
        # 2) QwenHTTPClient 默认 read timeout=120s 不够（生成大 JSON 慢），
        #    拉到 300s。
        max_tokens = max(2048, len(segments) * 400 + 512)
        result = llm_client.generate_json(
            prompt,
            system_prompt=_DIRECTOR_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            timeout=(10.0, 300.0),
        )
    except LLMError:
        return []
    except Exception:  # noqa: BLE001
        return []

    return _extract_directions(result, expected_ids=[s.segment_id for s in segments])


def _build_director_prompt(
    segments: list[Segment],
    characters: list[CharacterProfile],
    story_context: str,
) -> str:
    parts: list[str] = []
    if story_context:
        parts.append(f"## 故事上下文\n\n{story_context}")

    parts.append("\n## 角色档案\n")
    for c in characters:
        parts.append(
            f"- {c.name} ({c.role_type}/{c.gender or '?'}/{c.age_style or '?'}): "
            f"{c.personality or '无描述'}"
        )

    parts.append("\n## 需要指导的片段\n")
    for seg in segments:
        parts.append(
            f"- [{seg.segment_id}] speaker={seg.speaker}, type={seg.segment_type}\n"
            f"  text={seg.text}"
        )

    parts.append(
        "\n请为以上每个片段输出 segment_directions，数量必须等于输入片段数。"
        "delivery_instruction 必须结合原文内容，禁止空泛表达。"
    )
    return "\n".join(parts)


def _extract_directions(
    result: Any,
    *,
    expected_ids: list[str],
) -> list[dict[str, Any]]:
    """从 LLM 返回里抠出 segment_directions。

    兼容多种形状；只保留 segment_id 在 expected_ids 里的项。
    """
    raw_list: list[dict[str, Any]] = []
    if isinstance(result, dict):
        dirs = result.get("segment_directions")
        if isinstance(dirs, list):
            raw_list = [d for d in dirs if isinstance(d, dict)]
        elif "segment_id" in result:
            raw_list = [result]
    elif isinstance(result, list):
        raw_list = [d for d in result if isinstance(d, dict)]

    expected = set(expected_ids)
    return [d for d in raw_list if str(d.get("segment_id") or "") in expected]


def _build_instruction_from_llm(
    seg: Segment, raw: dict[str, Any]
) -> DirectorInstruction:
    """从 LLM 输出构造 DirectorInstruction，全部字段都做清洗。"""
    emotion = _clean_emotion(raw.get("emotion"))
    tone = _clean_tone(raw.get("tone"))
    volume = _clean_volume(raw.get("volume"))
    pitch = _clean_pitch(raw.get("pitch"))
    delivery = _clean_delivery(raw.get("delivery_instruction"))

    # delivery_instruction 如果 LLM 写得太空泛（只有 "自然对白语气" / "平稳叙述" 等），
    # 走 fallback 重写——比 LLM 给的更有内容。
    if not _delivery_has_content(delivery):
        fb = _fallback_instruction(seg, None)
        delivery = fb.delivery_instruction

    return DirectorInstruction(
        segment_id=seg.segment_id,
        speaker=seg.speaker,
        emotion=emotion,
        emotion_intensity=_clean_intensity(raw.get("emotion_intensity"),
                                           default=_default_intensity(emotion)),
        pace=_clean_pace(raw.get("pace")),
        tone=tone,
        volume=volume,
        pitch=pitch,
        pause_hint=_clean_pause(raw.get("pause_hint")),
        stress_words=_clean_stress_words(raw.get("stress_words")),
        delivery_instruction=delivery,
    )


# ── Fallback ───────────────────────────────────────────────────────────────

# narration 文本关键词 → 情绪映射（按优先级，前者命中就返回）
_NARRATION_EMOTION_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("nostalgic", ("故乡", "童年", "想起", "当年", "家乡", "回忆", "记忆", "往事", "怀念")),
    ("sad", ("悲伤", "难过", "失落", "惋惜", "遗憾", "哭泣", "眼泪")),
    ("warm", ("温暖", "温馨", "幸福", "母爱", "亲情", "慈爱")),
    ("joyful", ("快乐", "高兴", "喜欢", "开心", "欢喜", "兴奋", "雀跃")),
    ("anxious", ("担心", "忧虑", "害怕", "紧张", "不安")),
]


def _detect_narration_emotion(text: str) -> str:
    """根据文本关键词推断 narration 情绪。无命中 → calm。"""
    for emotion, keywords in _NARRATION_EMOTION_RULES:
        if any(kw in text for kw in keywords):
            return emotion
    return "calm"


def _detect_dialogue_emotion(text: str) -> str:
    """根据标点 / 关键词推断 dialogue 情绪。"""
    if any(pc in text for pc in ("？", "?")):
        return "surprised"
    if any(pc in text for pc in ("！", "!")):
        return "excited"
    if any(kw in text for kw in ("不要", "别", "不能")):
        return "serious"
    if any(kw in text for kw in ("想", "念", "记得")):
        return "longing"
    return "neutral"


# emotion → 默认 intensity（fallback 用）
_DEFAULT_INTENSITY_MAP: dict[str, float] = {
    "neutral": 0.3, "calm": 0.3,
    "gentle": 0.5, "warm": 0.55, "serious": 0.55,
    "nostalgic": 0.65, "longing": 0.65, "moved": 0.7,
    "playful": 0.7, "joyful": 0.75, "happy": 0.75,
    "excited": 0.85, "surprised": 0.75,
    "sad": 0.6, "anxious": 0.7,
    "angry": 0.85, "fearful": 0.8, "disgusted": 0.7,
}


def _default_intensity(emotion: str) -> float:
    return _DEFAULT_INTENSITY_MAP.get(emotion, 0.5)


# 各 emotion 对应的 narration fallback 完整字段
_NARRATION_FALLBACK_PROFILE: dict[str, dict[str, Any]] = {
    "nostalgic": dict(
        tone="warm", volume="soft", pitch="medium_low", pace=0.9, pause=0.8,
        delivery="以温和怀念的语气叙述，语速稍慢，突出画面感和思绪万千的氛围。",
    ),
    "joyful": dict(
        tone="lively", volume="normal", pitch="medium_high", pace=1.05, pause=0.6,
        delivery="以轻快愉悦的语气叙述，节奏明快，传递明亮的情绪色彩。",
    ),
    "warm": dict(
        tone="warm", volume="soft", pitch="medium", pace=0.95, pause=0.7,
        delivery="以温暖柔和的语气叙述，营造温馨的画面感。",
    ),
    "sad": dict(
        tone="serious", volume="soft", pitch="low", pace=0.85, pause=1.0,
        delivery="以低沉缓慢的语气叙述，传递感伤和沉重。",
    ),
    "anxious": dict(
        tone="serious", volume="soft", pitch="medium_high", pace=1.05, pause=0.6,
        delivery="以略带紧张压抑的语气叙述，节奏稍快，传递不安。",
    ),
    "calm": dict(
        tone="calm", volume="normal", pitch="medium", pace=0.95, pause=0.7,
        delivery="以平稳自然的语气叙述，节奏从容，画面感清晰。",
    ),
}


def _dialogue_fallback_for_child(text: str) -> dict[str, Any]:
    """童年视角的 dialogue fallback。"""
    if any(pc in text for pc in ("？", "?", "！", "!")):
        return dict(
            emotion="excited", tone="lively", volume="normal",
            pitch="medium_high", pace=1.15, pause=0.5,
            delivery="以童真兴奋的语气急切地说出，语速稍快，表现孩子的好奇或雀跃。",
        )
    return dict(
        emotion="playful", tone="lively", volume="normal",
        pitch="medium_high", pace=1.05, pause=0.5,
        delivery="以清亮天真的童声说出，语气活泼带好奇色彩。",
    )


def _dialogue_fallback_for_elderly(text: str) -> dict[str, Any]:
    """长辈视角的 dialogue fallback。"""
    if any(kw in text for kw in ("记得", "想起", "当年", "故乡")):
        return dict(
            emotion="nostalgic", tone="warm", volume="soft",
            pitch="medium_low", pace=0.88, pause=0.7,
            delivery="以沉稳怀念的口吻缓缓说出，语速偏慢，带岁月沉淀的厚重感。",
        )
    return dict(
        emotion="gentle", tone="warm", volume="soft",
        pitch="medium_low", pace=0.9, pause=0.6,
        delivery="以沉稳温和的长者口吻说出，语气中带着阅历和关怀。",
    )


def _dialogue_fallback_default(text: str) -> dict[str, Any]:
    """普通角色 dialogue fallback，按文本情绪推断。"""
    emotion = _detect_dialogue_emotion(text)
    if emotion == "excited":
        return dict(
            emotion=emotion, tone="lively", volume="normal",
            pitch="medium_high", pace=1.1, pause=0.5,
            delivery="以略带兴奋的语气急促说出，节奏明快。",
        )
    if emotion == "surprised":
        return dict(
            emotion=emotion, tone="lively", volume="normal",
            pitch="medium_high", pace=1.05, pause=0.5,
            delivery="以略带惊讶疑问的语气说出，句尾音调上扬。",
        )
    if emotion == "longing":
        return dict(
            emotion=emotion, tone="warm", volume="soft",
            pitch="medium_low", pace=0.92, pause=0.6,
            delivery="以柔和思念的语气说出，节奏稍缓，带眷恋色彩。",
        )
    if emotion == "serious":
        return dict(
            emotion=emotion, tone="serious", volume="normal",
            pitch="medium", pace=0.95, pause=0.6,
            delivery="以认真笃定的语气说出，节奏稳健。",
        )
    return dict(
        emotion="neutral", tone="normal", volume="normal",
        pitch="medium", pace=1.0, pause=0.5,
        delivery="以贴合台词情绪的自然语气说出。",
    )


def _fallback_instruction(
    seg: Segment, char: CharacterProfile | None
) -> DirectorInstruction:
    """LLM 没覆盖到该 segment 时的兜底，按 narrator / dialogue + 关键词给细粒度默认。"""
    if seg.segment_type in ("dialogue", "inner_thought"):
        if char and char.age_style == "child":
            profile = _dialogue_fallback_for_child(seg.text)
        elif char and char.age_style == "elderly":
            profile = _dialogue_fallback_for_elderly(seg.text)
        else:
            profile = _dialogue_fallback_default(seg.text)
        default_pause = 0.5
    else:
        emotion = _detect_narration_emotion(seg.text)
        profile = _NARRATION_FALLBACK_PROFILE.get(
            emotion, _NARRATION_FALLBACK_PROFILE["calm"]
        ).copy()
        profile["emotion"] = emotion
        default_pause = 0.7

    emotion = profile.get("emotion", "neutral")
    return DirectorInstruction(
        segment_id=seg.segment_id,
        speaker=seg.speaker,
        emotion=emotion,
        emotion_intensity=_default_intensity(emotion),
        pace=float(profile.get("pace", 1.0)),
        tone=str(profile.get("tone", "normal")),
        volume=str(profile.get("volume", "normal")),
        pitch=str(profile.get("pitch", "medium")),
        pause_hint=float(profile.get("pause", default_pause)),
        stress_words=[],  # fallback 不强行造词
        delivery_instruction=str(profile.get("delivery", "")),
    )


# ── 字段清洗 ────────────────────────────────────────────────────────────────

def _clean_emotion(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_EMOTIONS else "neutral"


def _clean_tone(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_TONES else "normal"


def _clean_volume(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_VOLUMES else "normal"


def _clean_pitch(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return s if s in _VALID_PITCHES else "medium"


def _clean_pace(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return _clamp(v, _PACE_MIN, _PACE_MAX)


def _clean_pause(raw: Any) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.7
    return _clamp(v, _PAUSE_MIN, _PAUSE_MAX)


def _clean_intensity(raw: Any, *, default: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return _clamp(v, _INTENSITY_MIN, _INTENSITY_MAX)


def _clean_stress_words(raw: Any) -> list[str]:
    """清洗 stress_words：必须 list[str]，最多 3 个，过滤空字符串。"""
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or len(s) > 12 or s in seen:
            continue
        seen.add(s)
        result.append(s)
        if len(result) >= _MAX_STRESS_WORDS:
            break
    return result


def _clean_delivery(raw: Any) -> str:
    s = str(raw or "").strip()
    if len(s) > _DELIVERY_MAX_LEN:
        s = s[:_DELIVERY_MAX_LEN]
    return s


# 空泛表达黑名单（命中即视为"无内容"，调 fallback 重写）
_GENERIC_DELIVERY_PHRASES = (
    "自然对白语气",
    "平稳叙述",
    "平稳叙述。",
    "平稳的叙述",
    "自然语气",
    "自然说话",
    "语气自然",
    "正常语气",
    "正常说话",
    "normal",
    "narration",
    "dialogue",
)


def _delivery_has_content(s: str) -> bool:
    """检查 delivery_instruction 是否有实质内容（不是空泛短语）。"""
    if not s or len(s) < _DELIVERY_MIN_LEN:
        return False
    lowered = s.lower().strip()
    if lowered in _GENERIC_DELIVERY_PHRASES:
        return False
    # 整句就是黑名单短语的轻微变体（前后多一两个标点）
    for phrase in _GENERIC_DELIVERY_PHRASES:
        if lowered == phrase or (len(lowered) <= len(phrase) + 3 and phrase in lowered):
            return False
    return True


# ── 通用工具 ────────────────────────────────────────────────────────────────

def _clamp(v: float, low: float, high: float) -> float:
    if v < low:
        return low
    if v > high:
        return high
    return v
