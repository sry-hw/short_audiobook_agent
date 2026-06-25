"""src_next/analysis/character_analyzer.py

角色档案生成（CharacterProfile 列表）。

数据流位置：
    Segment[]（已 resolve_speakers 处理过）
        → analyze_characters(segments, llm_client)
        → CharacterProfile[]（narrator 必在 index=0）

策略（v1 简化版）：
    1. 永远先放 narrator：用稳定 voice_prompt（参考旧 src/character_analyzer.py 的
       ``_NARRATOR_PROFILE``，精简成一句自然语言描述，供 Qwen VoiceGenerator 消费）。
    2. 从 segments 里按首次出现顺序收集所有非 narrator / 非 unknown 的 speaker。
    3. 把这些 speaker 一次性丢给 LLM 生成档案。
    4. LLM 返回结构异常（包括 MockLLM 的占位 dict）→ 每个 speaker 走 fallback
       （根据名字关键词猜动物 / 老人 / 儿童，给一个低 confidence 档案）。
    5. 单角色失败不阻塞其他角色。

参考旧 src/character_analyzer.py：
- narrator 硬编码，不走 LLM；
- voice_instruction 用自然语言一句话描述，便于后续 TTS 直接消费；
- 角色 prompt 列出 gender / age / timbre / role_type / confidence 字段。

本层与旧 src 的差异：
- 字段从旧版 dict 改为 ``src_next.core.data_models.CharacterProfile`` dataclass；
- ``timbre`` 字段并入 voice_prompt 自然语言描述，不再单独保留；
- ``voice_prompt`` 强约束 ``用...说`` 格式，便于透传到 Qwen VoiceGenerator 的 ``--instruct``；
- 不再生成 ``reason`` 字段（debug 时看 LLM 原始日志即可）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.data_models import CharacterProfile, Segment
from ..llm.base import BaseLLMClient, LLMError


# ── narrator 固定档案 ──────────────────────────────────────────────────────

_NARRATOR_VOICE_PROMPT = "温柔亲切的年轻女性声音，语气平稳，富有讲故事的感觉"


def _default_narrator() -> CharacterProfile:
    return CharacterProfile(
        name="narrator",
        role_type="narrator",
        gender="female",
        age_style="young",
        personality="温柔亲切，平稳客观",
        voice_prompt=_NARRATOR_VOICE_PROMPT,
        confidence=0.95,
    )


# ── 安全称谓后缀（fallback 用） ─────────────────────────────────────────────

# 安全后缀剥离白名单（≥2 字、语义明确的社会称谓，剥离零风险）。
# 仅用于 LLM 没返回 alias_map 时的 fallback；不与 story_resolver 共享
# （story_resolver 走最保守路径，不做后缀剥离）。
_SAFE_TITLES = (
    "先生", "女士", "老师", "师傅", "同志", "小姐",
    "大叔", "阿姨", "爷爷", "奶奶",
)


# ── LLM prompt ─────────────────────────────────────────────────────────────

# ── narrator 系统 prompt（独立调用，基于全文判断文体风格） ────────────────

_NARRATOR_SYSTEM_PROMPT = """你是一个中文有声书旁白音色分析师。基于故事全文判断文体风格，推荐最合适的旁白音色。

严格输出 JSON，结构如下：

{
  "gender": "male / female",
  "age_style": "young / middle_aged / elderly",
  "personality": "一句话描述朗读风格",
  "voice_prompt": "Qwen3-TTS VoiceDesign instruction 格式的描述性短语",
  "genre": "fairy_tale / wuxia / romance / mystery / essay / knowledge / other",
  "confidence": 0.0~1.0
}

文体风格 → 旁白音色参考：

- **fairy_tale**（童话 / 儿童故事 / 寓言）：温暖柔和的青年女性声音，语气亲切，适合讲故事
- **wuxia**（武侠 / 历史 / 古风）：沉稳磁性的中年男性声音，语速偏慢，富有画面感
- **romance**（都市言情 / 言情）：温柔细腻的青年女性声音，情感饱满
- **mystery**（悬疑 / 推理 / 恐怖）：低沉冷静的中年男性声音，语调平稳，富有悬念感
- **essay**（散文 / 抒情 / 随笔）：平缓柔和的青年女性声音，富有诗意
- **knowledge**（科普 / 说明 / 教程）：专业清晰的青年声音，吐字准确，逻辑感强
- **other**：根据文本整体基调灵活判断

字段要求：

- voice_prompt 必须是 Qwen3-TTS VoiceDesign 期望的自然语言描述性短语（**不是「用...说」格式**），长度 10~50 字
- 必须明确包含：性别 + 年龄感 + 音色特征 + 朗读风格
- 参考写法：
  * "温柔亲切的年轻女性声音，语气平稳，富有讲故事的感觉"
  * "沉稳磁性的中年男性声音，语速缓慢，富有画面感"
  * "低沉冷静的中年男性声音，语调平稳，适合悬疑叙事"
- genre 用于人工排错，不影响下游
- confidence 反映对文体判断的把握程度
"""


# ── LLM prompt ─────────────────────────────────────────────────────────────

_CHARACTER_SYSTEM_PROMPT = """你是一个中文故事角色声音分析师。根据故事全文和每个角色的台词，为角色生成声音档案，并合并指代同一角色的别名。

严格输出 JSON，结构如下：

{
  "characters": [
    {
      "name": "canonical 角色名（优先采用原文首次出现且稳定的称呼）",
      "aliases": ["别名1", "别名2"],
      "gender": "male / female",
      "age_style": "child / young / middle_aged / elderly",
      "personality": "一句话描述角色性格",
      "voice_prompt": "用...的嗓音说（开头必须是'用'，结尾必须是'说'）",
      "confidence": 0.0~1.0
    }
  ],
  "alias_map": {
    "别名": "canonical 名"
  }
}

归并规则（**非常重要**）：
1. 同一个角色的不同称呼必须合并到一条 character 记录中，canonical 名优先采用原文首次出现且稳定的称呼（优先级：人名 > 职务 > 代词）。
2. aliases 数组列出所有指代该角色的别名（不含 canonical 本身）。
3. alias_map 必须把每个别名映射到 canonical 名；alias_map 的 value 必须出现在 characters[].name 中。
4. 不要凭常识替换原文称呼：原文叫"小明"就不要 canonical 改成"明"；原文叫"豆豆"就不要改成"小豆豆"。
5. 不同角色不得合并：父子关系（"小明" vs "小明爸爸"）、主仆关系、人和动物均不得当作同一角色。
6. narrator 永远独立，不在 characters 数组中。

字段要求：
- voice_prompt 必须是 Qwen3-TTS VoiceDesign 期望的自然语言描述性短语（**不是「用...说」格式**，usage_guide_qwen3.md 没有此要求）
- 长度 10~50 字
- 必须明确包含以下 4 个维度，避免模型自由发挥导致性别错乱：
  * 性别（男 / 女 / 中性）
  * 年龄感（童 / 青年 / 中年 / 老年）
  * 音色特征（清亮 / 沙哑 / 磁性 / 柔和 / 苍老 / 清脆 ...）
  * 情绪或语气（亲切 / 严肃 / 活泼 / 温暖 / 平稳 ...）
- 参考写法（Qwen3-TTS 官方示例风格）：
  * "温柔关切的中年女性声音，语速平稳，带着母爱的温暖"
  * "清亮活泼的小女孩声音，天真可爱，语调上扬"
  * "低沉磁性的中年男性声音，语速缓慢，成熟稳重"
  * "苍老沙哑的老年声音，语速偏慢，富有沧桑感"
  * "专业清晰的青年女性声音，吐字准确，富有讲故事的感觉"
- **禁止**输出「用...嗓音说」格式（会让 Qwen3-TTS 模型困惑，可能返回错误性别）
- 性别不确定时优先标 confidence<0.6
- 不能确定的角色也必须输出（confidence 标低），不能省略
- characters 数组长度可以小于输入的 speaker 数（因为合并）；但每个输入 speaker 都必须被覆盖：要么作为 character.name，要么作为 aliases 中的别名，要么在 alias_map 中作为 key。
"""


# ── 动物 / 老人 / 儿童 关键词（fallback 用） ────────────────────────────────

_ANIMAL_KEYWORDS = (
    "松鼠", "兔子", "狐狸", "猫", "狗", "熊", "虎", "狮", "狼", "鹿",
    "猴", "鸡", "鸭", "鹅", "鱼", "龟", "乌鸦", "鸟", "蚱蜢", "蚂蚁",
    "蝴蝶", "蜜蜂", "龙", "蛇", "马", "牛", "羊", "猪", "鼠", "大象",
    "老鼠", "乌龟", "鹦鹉", "燕子", "麻雀", "喜鹊", "青蛙", "螃蟹",
)

_ELDERLY_KEYWORDS = ("老", "爷爷", "奶奶", "公公", "婆婆", "大叔", "大婶", "先生")

_CHILD_KEYWORDS = ("小宝宝", "宝宝", "孩", "童", "弟弟", "妹妹", "小男孩", "小女孩")


# ── 入口函数 ────────────────────────────────────────────────────────────────

def analyze_characters(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[CharacterProfile]:
    """从 resolved segments 生成 CharacterProfile 列表（含别名归并）。

    Args:
        segments: 已经跑过 ``resolve_speakers`` 的 Segment 列表（speaker 已填充）。
        llm_client: 实现 ``BaseLLMClient`` 的任意后端。
        story_context: 故事全文（推荐传入全文而非仅标题，让 LLM 做更准的
            别名归并判断）。文本长度 ≤3000 字时直接传全文。

    Returns:
        CharacterProfile 列表。narrator 永远在 index=0；其余按 canonical name
        在 segments 中首次出现位置排序。同一角色的多个称呼已合并到一个
        CharacterProfile，被合并的称呼写进 ``aliases`` 字段。单角色 LLM
        失败时走 fallback，不会丢角色。
    """
    # narrator 走 LLM 动态分析（基于全文判断文体风格 → 推荐旁白音色）；
    # 失败 / 全文太短 / 结构异常 → fallback 到默认年轻女声。
    narrator = _analyze_narrator_via_llm(story_context, llm_client) or _default_narrator()

    unique_speakers = _extract_unique_speakers(segments)
    if not unique_speakers:
        return [narrator]

    profiles, alias_map = _analyze_via_llm(
        unique_speakers, segments, llm_client, story_context,
    )

    # LLM 没给 alias_map（结构异常 / MockLLMClient）→ 安全规则兜底
    if not alias_map:
        alias_map = _build_alias_map_from_safe_rules(unique_speakers)

    merged = _merge_speakers(unique_speakers, alias_map, profiles)
    ordered = _sort_by_first_appearance(merged, segments)
    return [narrator] + ordered


# ── 内部工具 ────────────────────────────────────────────────────────────────

def _extract_unique_speakers(segments: list[Segment]) -> list[str]:
    """按首次出现顺序收集 speaker，排除 narrator / unknown / 空。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for seg in segments:
        name = (seg.speaker or "").strip()
        if not name or name in ("narrator", "unknown"):
            continue
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _analyze_via_llm(
    speakers: list[str],
    segments: list[Segment],
    llm_client: BaseLLMClient,
    story_context: str,
) -> tuple[dict[str, CharacterProfile], dict[str, str]]:
    """调用 LLM 一次性分析所有 speaker，返回 (profiles, alias_map)。

    LLM 失败 / 结构异常时返回 ``({}, {})``，让上层走 fallback。
    """
    if not speakers:
        return {}, {}

    prompt = _build_character_prompt(speakers, segments, story_context)
    try:
        result = llm_client.generate_json(
            prompt, system_prompt=_CHARACTER_SYSTEM_PROMPT
        )
    except LLMError:
        return {}, {}
    except Exception:  # noqa: BLE001
        return {}, {}

    return _extract_character_profiles(result)


def _build_character_prompt(
    speakers: list[str],
    segments: list[Segment],
    story_context: str,
) -> str:
    parts: list[str] = []
    if story_context:
        parts.append(f"## 故事全文（请通读后再判断角色合并）\n\n{story_context}")

    # 每个 speaker 收集前 3 条代表性台词，给 LLM 判断素材
    lines_by_speaker: dict[str, list[str]] = {name: [] for name in speakers}
    for seg in segments:
        name = (seg.speaker or "").strip()
        if name in lines_by_speaker and len(lines_by_speaker[name]) < 3:
            lines_by_speaker[name].append(seg.text)

    parts.append("\n## 角色与台词（按首次出现顺序）\n")
    for name in speakers:
        parts.append(f"### {name}")
        for line in lines_by_speaker[name]:
            parts.append(f"- {line}")
        if not lines_by_speaker[name]:
            parts.append("- （无对白）")

    parts.append(
        "\n请基于全文判断 speaker 之间是否存在别名关系，"
        "把指代同一角色的 speaker 合并到 canonical 名（characters[].name），"
        "并在 alias_map 中列出「别名 → canonical 名」映射。"
        "characters 数组长度可以小于输入 speaker 数，但每个 speaker 必须被覆盖。"
    )
    return "\n".join(parts)


def _extract_character_profiles(
    result: Any,
) -> tuple[dict[str, CharacterProfile], dict[str, str]]:
    """从 LLM 返回里抠出 (profiles, alias_map)。

    兼容：
    * ``{"characters": [...], "alias_map": {...}}``   ← 期望
    * ``{"characters": [...]}``                       ← 没 alias_map
    * ``[{"name": ...}, ...]``                        ← 旧 list 形式
    * ``{"source": "mock", ...}``                     ← MockLLMClient 默认 dict
    """
    raw_list, alias_map = _parse_llm_output(result)

    profiles: dict[str, CharacterProfile] = {}
    for raw in raw_list:
        name = str(raw.get("name") or "").strip()
        if not name or name == "narrator":
            continue
        profiles[name] = _build_profile_from_llm(name, raw)
    return profiles, alias_map


def _parse_llm_output(result: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """解析 LLM 输出为 (characters_raw_list, alias_map)。

    alias_map 缺失 / 类型异常时返回 ``{}``，让上层走 fallback。
    """
    raw_list: list[dict[str, Any]] = []
    alias_map: dict[str, str] = {}

    if isinstance(result, dict):
        chars = result.get("characters")
        if isinstance(chars, list):
            raw_list = [c for c in chars if isinstance(c, dict)]
        elif "name" in result:
            # dict 看起来像单条 character
            raw_list = [result]
        am = result.get("alias_map")
        if isinstance(am, dict):
            alias_map = {
                str(k): str(v)
                for k, v in am.items()
                if isinstance(k, str) and isinstance(v, str)
                and str(k).strip() and str(v).strip()
            }
    elif isinstance(result, list):
        raw_list = [c for c in result if isinstance(c, dict)]

    return raw_list, alias_map


def _build_profile_from_llm(name: str, raw: dict[str, Any]) -> CharacterProfile:
    """从单条 LLM 输出构造 CharacterProfile，字段缺失时用合理默认。"""
    gender = _clean_str(raw.get("gender"), valid={"male", "female"})
    age_style = _clean_str(
        raw.get("age_style"),
        valid={"child", "young", "middle_aged", "elderly"},
    )
    personality = str(raw.get("personality") or "").strip() or None
    voice_prompt = _clean_voice_prompt(raw.get("voice_prompt"), name)
    confidence = _clean_confidence(raw.get("confidence"), default=0.6)

    # aliases：LLM 可能漏字段 / 误填 canonical 名 / 带空字符串，全部清洗
    aliases_raw = raw.get("aliases") or []
    aliases_clean: list[str] = []
    seen: set[str] = set()
    for a in aliases_raw:
        if not isinstance(a, str):
            continue
        a_str = a.strip()
        if not a_str or a_str == name or a_str in seen:
            continue
        seen.add(a_str)
        aliases_clean.append(a_str)

    return CharacterProfile(
        name=name,
        role_type="character",
        gender=gender,
        age_style=age_style,
        personality=personality,
        voice_prompt=voice_prompt,
        confidence=confidence,
        aliases=aliases_clean,
    )


def _clean_voice_prompt(raw: Any, name: str) -> str:
    """voice_prompt 校验：必须是 Qwen3-TTS 期望的描述性短语，长度 8~60 字。

    不再强制「用...说」格式（usage_guide_qwen3.md 没有此要求，反而会让
    Qwen3-TTS 模型困惑导致性别错乱）。只要长度合理就放过，让 LLM 输出
    的自然语言描述直接透传。长度不达标 / 空字符串走 fallback。
    """
    s = str(raw or "").strip()
    if 8 <= len(s) <= 60:
        return s
    return _fallback_voice_prompt(name)


def _fallback_voice_prompt(name: str) -> str:
    """根据名字关键词生成兜底 voice_prompt（Qwen3-TTS 描述性短语格式）。

    判断顺序：老人 > 儿童 > 动物 > 默认。
    顺序很关键：``老乌龟`` 同时命中 ``老``（老人）和 ``乌龟``（动物），
    老人优先才能拿到沉稳老者嗓音。
    """
    if any(kw in name for kw in _ELDERLY_KEYWORDS):
        return "沉稳温暖的中老年男性声音，语速偏慢，富有沧桑感"
    if any(kw in name for kw in _CHILD_KEYWORDS):
        return "清亮活泼的童声，天真可爱，语调上扬"
    if any(kw in name for kw in _ANIMAL_KEYWORDS):
        # 小动物默认偏儿童感 / female 倾向（儿童故事常见设定）
        return "清亮活泼的童声，天真可爱，富有故事感"
    return "自然真实的人声，语速平稳，吐字清晰"


def _analyze_narrator_via_llm(
    story_context: str,
    llm_client: BaseLLMClient,
) -> CharacterProfile | None:
    """让 LLM 基于全文判断文体风格，生成 narrator 档案。

    失败 / 全文太短 / 返回结构异常 → 返回 None，让上层 fallback 到
    ``_default_narrator``。narrator 不走 character_analyzer 的 alias_map
    合并路径，单独调一次 LLM。
    """
    if not story_context or len(story_context.strip()) < 20:
        return None

    prompt = (
        f"## 故事全文\n\n{story_context}\n\n"
        "请基于全文判断文体风格，推荐最合适的旁白音色。"
    )
    try:
        result = llm_client.generate_json(
            prompt, system_prompt=_NARRATOR_SYSTEM_PROMPT
        )
    except LLMError:
        return None
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(result, dict):
        return None

    gender = _clean_str(result.get("gender"), valid={"male", "female"})
    age_style = _clean_str(
        result.get("age_style"),
        valid={"child", "young", "middle_aged", "elderly"},
    )
    personality = str(result.get("personality") or "").strip() or None
    voice_prompt = _clean_voice_prompt(result.get("voice_prompt"), "narrator")
    confidence = _clean_confidence(result.get("confidence"), default=0.7)

    return CharacterProfile(
        name="narrator",
        role_type="narrator",
        gender=gender or "female",
        age_style=age_style or "young",
        personality=personality or "温柔亲切，平稳客观",
        voice_prompt=voice_prompt,
        confidence=confidence,
    )


def _fallback_character_profile(name: str) -> CharacterProfile:
    """LLM 完全没覆盖到该角色时的最终 fallback。

    判断顺序同 ``_fallback_voice_prompt``：老人 > 儿童 > 动物 > 默认。
    """
    if any(kw in name for kw in _ELDERLY_KEYWORDS):
        return CharacterProfile(
            name=name,
            role_type="character",
            gender="male",
            age_style="elderly",
            personality="沉稳",
            voice_prompt="用沉稳温暖的老者嗓音说",
            confidence=0.4,
        )
    if any(kw in name for kw in _CHILD_KEYWORDS):
        return CharacterProfile(
            name=name,
            role_type="character",
            gender="female",
            age_style="child",
            personality="天真",
            voice_prompt="用天真可爱的童声说",
            confidence=0.4,
        )
    if any(kw in name for kw in _ANIMAL_KEYWORDS):
        return CharacterProfile(
            name=name,
            role_type="character",
            gender="female",
            age_style="child",
            personality="活泼",
            voice_prompt="用清亮活泼的童声说",
            confidence=0.4,
        )
    return CharacterProfile(
        name=name,
        role_type="character",
        gender=None,
        age_style=None,
        personality=None,
        voice_prompt="用自然真实的人声说",
        confidence=0.3,
    )


def _clean_str(raw: Any, *, valid: set[str]) -> str | None:
    s = str(raw or "").strip().lower()
    return s if s in valid else None


def _clean_confidence(raw: Any, *, default: float) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


# ── 角色归并（merge + sort + fallback） ─────────────────────────────────────

def _merge_speakers(
    speakers: list[str],
    alias_map: dict[str, str],
    profiles: dict[str, CharacterProfile],
) -> list[CharacterProfile]:
    """按 alias_map 把别名 speaker 折叠到 canonical，累积 aliases。

    对于每个 canonical：
    - 优先用 LLM 给的 profile（按 canonical 名命中）
    - 否则按别名在 profiles 里找
    - 都没有 → fallback profile
    - 把所有指向该 canonical 的 speaker（含 speakers 里被 alias_map
      重定向的，也含 LLM 直接写在 characters[].aliases 里的）合并到
      ``aliases`` 字段（去重，不含 canonical 本身）。
    """
    canonical_to_speakers: dict[str, list[str]] = defaultdict(list)
    for sp in speakers:
        canon = alias_map.get(sp, sp)
        canonical_to_speakers[canon].append(sp)

    result: list[CharacterProfile] = []
    for canon, sp_list in canonical_to_speakers.items():
        if canon == "narrator":
            # 防御性：narrator 不应出现在 speakers 里，跳过
            continue

        # 选 profile：canonical 名优先；否则任一别名命中；都没有走 fallback
        prof = profiles.get(canon)
        if prof is None:
            for sp in sp_list:
                if sp in profiles:
                    prof = profiles[sp]
                    break
        if prof is None:
            prof = _fallback_character_profile(canon)

        # 累积 aliases：LLM profile 自带的 + speakers 里被合并到 canon 的
        all_aliases: list[str] = []
        seen: set[str] = set()
        for a in (prof.aliases or []):
            if a != canon and a not in seen:
                seen.add(a)
                all_aliases.append(a)
        for sp in sp_list:
            if sp != canon and sp not in seen:
                seen.add(sp)
                all_aliases.append(sp)

        # 用 prof 的副本，覆盖 aliases；name 强制设为 canon 保持一致
        prof.aliases = all_aliases
        prof.name = canon
        result.append(prof)

    return result


def _sort_by_first_appearance(
    profiles: list[CharacterProfile],
    segments: list[Segment],
) -> list[CharacterProfile]:
    """按 canonical name 在 segments 中首次出现位置排序。

    匹配规则：``segment.speaker == profile.name`` 或 ``segment.speaker``
    在 ``profile.aliases`` 中。未匹配到的角色排到最后（按原顺序稳定）。
    """
    first_idx: dict[str, tuple[int, str]] = {}
    for seg in segments:
        spk = (seg.speaker or "").strip()
        if not spk:
            continue
        for p in profiles:
            if spk == p.name or spk in (p.aliases or []):
                if p.name not in first_idx:
                    first_idx[p.name] = (seg.raw_index, seg.segment_id)

    def key(p: CharacterProfile) -> tuple[float, str]:
        return first_idx.get(p.name, (float('inf'), ""))

    return sorted(profiles, key=key)


def _build_alias_map_from_safe_rules(speakers: list[str]) -> dict[str, str]:
    """LLM 没给 alias_map 时的保守兜底。

    只做安全后缀剥离：speaker 以 ``_SAFE_TITLES`` 中的某个后缀结尾，
    且剥离后剩下的部分（≥2 字）能命中 speakers 列表中的另一个名字，
    才合并。其他情况一律不合并。
    """
    if not speakers:
        return {}

    known = set(speakers)
    am: dict[str, str] = {}
    for sp in speakers:
        for title in _SAFE_TITLES:
            if sp.endswith(title) and len(sp) > len(title):
                core = sp[:-len(title)]
                if len(core) >= 2 and core in known and core != sp:
                    am[sp] = core
                    break
    return am
