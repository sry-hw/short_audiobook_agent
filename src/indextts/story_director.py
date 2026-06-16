"""IndexTTS 版故事导演层：直接输出 8 维情绪向量参数。"""

import json
import os
from pathlib import Path
from typing import Dict

import requests

_SYSTEM_PROMPT = """你是一个有声书导演，直接为 IndexTTS 生成 8 维情绪向量参数。

严格输出JSON，不要输出其他内容。结构如下：

{
  "overall_style": {
    "genre": "故事类型（如散文/寓言/童话/历史故事）",
    "tone": "整体基调（如温暖怀旧/紧张激烈/轻松幽默）",
    "pace": "整体节奏（如舒缓/适中/紧凑）",
    "summary": "一句话概括故事氛围"
  },
  "characters_direction": {
    "角色名": {
      "voice_direction": "对这个角色声音表现的指导",
      "performance_note": "表演要点，角色性格在台词中的体现"
    }
  },
  "segment_directions": [
    {
      "segment_id": 1,
      "emotion_vector": [0, 0, 0.1, 0, 0, 0, 0.1, 0.8],
      "emotion_text": "用平静略带起伏的语气说",
      "interval_silence": 500,
      "delivery_note": "旁白叙述，节奏舒缓",
      "emphasis_words": ["松鼠", "松果"]
    }
  ]
}

## 8 维情绪向量说明

向量共 8 维，索引对应关系如下：
- 索引 0：高兴（happiness）
- 索引 1：愤怒（anger）
- 索引 2：悲伤（sad）
- 索引 3：恐惧（fear）
- 索引 4：反感（disgust）
- 索引 5：低落（melancholic）
- 索引 6：惊讶（surprise）
- 索引 7：平静/自然（calm）

**约束：各维之和 ≤ 0.8**

## 向量设计原则

- 平静叙述：第7维（平静）0.8-1.0，其他维接近 0
- 开心/兴奋：第0维（高兴）为主（0.3-0.6），可配合第6维（惊讶）0.1-0.2
- 担忧：第3维（恐惧）为主（0.3-0.5），配合第5维（低落）0.1-0.2
- 愤怒：第1维（愤怒）为主（0.3-0.6）
- 悲伤：第2维（悲伤）为主（0.3-0.6），配合第5维（低落）0.1-0.2
- 坚定：第7维（平静）高（0.5-0.7），少量第1维（愤怒）0.1-0.2
- 惊讶：第6维（惊讶）为主（0.4-0.7）
- 所有向量第7维（平静）至少保留 0.1 作为基底，避免完全非自然状态

## interval_silence 设计

- 段落结尾长停：500-800ms
- 段内短停：200-400ms
- 戏剧性时刻：800-1200ms
- 旁白转对白过渡：300-500ms

## 判断依据

- 旁白：回忆性叙述偏平静柔和，描写性叙述偏平稳
- 对白：根据角色性格和台词内容判断情绪向量
- delivery_note：一句话表演指导
- emphasis_words：只选 1-3 个真正需要重读的词，不要多选

重要：segment_directions 必须覆盖输入的每一个 segment，不能遗漏。"""


def direct_story(
    segments: Dict,
    characters: Dict,
    story_text: str = "",
) -> Dict:
    """生成完整导演计划（IndexTTS 专用参数格式）。"""
    user_prompt = _build_user_prompt(segments, characters, story_text)
    env = _load_env()
    raw = _call_llm(_SYSTEM_PROMPT, user_prompt, env)
    directing = _parse_response(raw)
    directing = _validate_and_fill(directing, segments)
    return directing


def _build_user_prompt(
    segments: Dict, characters: Dict, story_text: str
) -> str:
    parts = []
    if story_text:
        parts.append("## 故事全文\n\n" + story_text)

    parts.append("\n## 角色信息\n")
    n = characters.get("narrator", {})
    parts.append(f"- narrator: {n.get('gender', '')}, {n.get('age', '')}, {n.get('timbre', '')}")
    for c in characters.get("characters", []):
        parts.append(f"- {c['speaker']}: {c['gender']}, {c['age']}, {c['timbre']} ({c['role_type']})")

    parts.append("\n## 需要指导的片段\n")
    for seg in segments["segments"]:
        parts.append(f"[seg_{seg['segment_id']}] ({seg['speaker']}, {seg['type']}) {seg['text']}")

    parts.append("\n请为以上每个片段生成 8 维情绪向量和表演指导，严格输出JSON。")
    return "\n".join(parts)


def _load_env() -> Dict[str, str]:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    config = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def _call_llm(system_prompt: str, user_prompt: str, env: Dict[str, str]) -> str:
    base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
    api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
    model = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", "qwen3.6-plus"))

    if not base_url or not api_key:
        raise ValueError("缺少 LLM 配置。请在 .env 文件中设置 LLM_BASE_URL 和 LLM_API_KEY。")

    base = base_url.rstrip("/")
    url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.2,
    }

    response = requests.post(
        url, headers=headers, json=payload, timeout=(10, 180), verify=False
    )
    response.raise_for_status()
    data = response.json()
    for block in data["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"LLM 响应中没有 text block: {data}")


def _parse_response(raw: str) -> Dict:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return _fallback_director_plan(text)
    text = text[first_brace:last_brace + 1]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        debug_dir = Path(__file__).resolve().parent.parent.parent / "output" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        raw_path = debug_dir / f"indextts_director_raw_{ts}.txt"
        clean_path = debug_dir / f"indextts_director_cleaned_{ts}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        clean_path.write_text(text, encoding="utf-8")
        pos = e.pos
        snippet = text[max(0, pos - 100):min(len(text), pos + 100)]
        print(f"\n[WARN] JSON 解析失败 at pos={pos}, saved to {raw_path.name}")
        print(f"  Context: ...{snippet}...")
        print(f"  Error: {e.msg}")
        print(f"  → 返回简化 fallback plan")
        return _fallback_director_plan(text)


def _fallback_director_plan(raw_text: str) -> Dict:
    genre = "未知"
    for kw in ["寓言", "童话", "儿童故事", "历史故事", "散文", "小说", "科普"]:
        if kw in raw_text[:500]:
            genre = kw
            break
    return {
        "overall_style": {"genre": genre, "tone": "中性", "pace": "正常", "summary": "（LLM JSON 解析失败，使用默认导演计划）"},
        "characters_direction": {},
        "segment_directions": [],
        "_parse_failed": True,
    }


def _validate_and_fill(directing: Dict, segments: Dict) -> Dict:
    if "overall_style" not in directing:
        directing["overall_style"] = {"genre": "未知", "tone": "中性", "pace": "适中", "summary": ""}

    if "characters_direction" not in directing:
        directing["characters_direction"] = {}

    if "narrator" not in directing["characters_direction"]:
        directing["characters_direction"]["narrator"] = {"voice_direction": "平稳自然的旁白", "performance_note": ""}

    seg_dirs = directing.get("segment_directions", [])
    dir_map = {d["segment_id"]: d for d in seg_dirs}

    filled = []
    for seg in segments["segments"]:
        sid = seg["segment_id"]
        if sid in dir_map:
            d = dict(dir_map[sid])
            d.setdefault("emotion_vector", [0, 0, 0, 0, 0, 0, 0, 1.0])
            d.setdefault("emotion_text", "")
            d.setdefault("interval_silence", 400)
            d.setdefault("delivery_note", "")
            d.setdefault("emphasis_words", [])
            # 校验并修正 emotion_vector
            d["emotion_vector"] = _normalize_vector(d["emotion_vector"])
        else:
            d = {
                "segment_id": sid,
                "emotion_vector": [0, 0, 0, 0, 0, 0, 0, 1.0],
                "emotion_text": "",
                "interval_silence": 400,
                "delivery_note": "",
                "emphasis_words": [],
            }
        d["speaker"] = seg["speaker"]
        d["text"] = seg["text"]
        filled.append(d)

    directing["segment_directions"] = filled
    return directing


def _normalize_vector(vec: list) -> list:
    """校验并修正 8 维情绪向量：补零/截断/缩放。"""
    if not isinstance(vec, (list, tuple)):
        return [0, 0, 0, 0, 0, 0, 0, 1.0]
    # 补零或截断到 8 维
    vec = list(vec)
    if len(vec) < 8:
        vec += [0] * (8 - len(vec))
    elif len(vec) > 8:
        vec = vec[:8]
    # 确保第7维（平静）至少 0.1
    if vec[7] < 0.1:
        vec[7] = 0.1
    total = sum(vec)
    if total > 0.8:
        scale = 0.8 / total
        vec = [v * scale for v in vec]
    return vec