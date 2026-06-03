"""角色声音分析器：从 segments 中提取角色，用 LLM 分析声音特征。

输出三维声音标签（gender/age/timbre），为后续 TTS 选音色提供依据。
narrator 硬编码，不走 LLM。
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import requests

_NARRATOR_PROFILE = {"gender": "female", "age": "young", "timbre": "warm"}

_SYSTEM_PROMPT = """你是一个中文故事的角色声音分析师。根据故事内容和角色的台词，为每个说话人确定声音特征。

输出严格的JSON，不要输出其他内容。每个角色包含以下字段：
- speaker: 说话人名称
- gender: male / female
- age: child / young / middle_aged / elderly
- timbre: bright / warm / deep / soft / sharp / rough
- role_type: protagonist / supporting / minor
- confidence: high / medium / low
- reason: 一句话说明判断依据

判断依据：
- 性别和年龄从角色名称和故事内容推断（如"母亲"→ female + middle_aged，"我"结合上下文判断）
- 音色从角色的言行举止推断（如温柔说话 → warm，怒气冲冲 → sharp/rough）
- 如无法确定，优先选择最可能的值，confidence 标记为 low

输出格式：
{"characters":[{"speaker":"母亲","gender":"female","age":"middle_aged","timbre":"warm","role_type":"supporting","confidence":"high","reason":"母亲形象，说话关心家人"}]}"""


def analyze_characters(segments: Dict, story_text: str = "") -> Dict:
    """从 segments 分析角色声音特征。

    Args:
        segments: segment_builder.build_segments() 的输出
        story_text: 故事全文（提供给 LLM 作为上下文）

    Returns:
        包含 characters 列表和 narrator 配置的字典
    """
    speaker_lines = _extract_speakers(segments["segments"])

    if not speaker_lines:
        return {
            "characters": [],
            "narrator": dict(_NARRATOR_PROFILE),
        }

    user_prompt = _build_user_prompt(speaker_lines, story_text)
    env = _load_env()
    raw = _call_llm(_SYSTEM_PROMPT, user_prompt, env)
    characters = _parse_response(raw)

    return {
        "characters": characters,
        "narrator": dict(_NARRATOR_PROFILE),
    }


def _extract_speakers(segments: List[Dict]) -> Dict[str, List[str]]:
    """从 segments 中提取所有唯一 speaker（排除 narrator/unknown）及其台词。"""
    result: Dict[str, List[str]] = {}
    for seg in segments:
        speaker = seg.get("speaker", "")
        if speaker in ("narrator", "unknown", ""):
            continue
        if seg.get("type") != "dialogue":
            continue
        result.setdefault(speaker, []).append(seg["text"])
    return result


def _build_user_prompt(speaker_lines: Dict[str, List[str]], story_text: str) -> str:
    """构建 LLM 的 user prompt。"""
    parts = []
    if story_text:
        parts.append("## 故事全文\n\n" + story_text)

    parts.append("## 角色台词\n")
    for speaker, lines in speaker_lines.items():
        parts.append(f"### {speaker}")
        for line in lines:
            parts.append(f"- {line}")

    parts.append("\n请为以上每个角色分析声音特征，严格输出JSON。")
    return "\n".join(parts)


def _load_env() -> Dict[str, str]:
    """从 .env 文件加载配置。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    config = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def _call_llm(system_prompt: str, user_prompt: str, env: Dict[str, str]) -> str:
    """调用 LLM API（Anthropic Messages 格式）。"""
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
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.1,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=(10, 120), verify=False)
    response.raise_for_status()

    data = response.json()
    for block in data["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"LLM 响应中没有 text block: {data}")


def _parse_response(raw: str) -> List[Dict]:
    """解析 LLM 返回的 JSON。"""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    parsed = json.loads(text.strip())
    if "characters" in parsed:
        return parsed["characters"]
    return parsed if isinstance(parsed, list) else []
