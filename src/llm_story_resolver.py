"""LLM 语义判断层：识别引号类型和说话人。

接收 story_parser 的输出，逐段落调用 LLM 对 quoted part 做语义判断。
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import requests

# ── quote_type 分类定义 ───────────────────────────────────────
# dialogue      真实说出口的话
# inner_thought 心理活动、心想的话
# quoted_term   强调词、概念、特殊称谓
# title_or_name 书名、篇名、故事名、人名引用
# unknown       无法判断

_SYSTEM_PROMPT = """判断中文故事中每个引号的类型和说话人。严格输出JSON，不要输出其他内容。

quote_type: dialogue(对话) / inner_thought(心理活动) / quoted_term(强调词概念) / title_or_name(书名篇名) / unknown
speaker: dialogue或inner_thought填说话人(不确定填"unknown")，其他填null
confidence: high / medium / low
reason: 一句话说明依据

输出格式：
{"resolutions":[{"part_id":"p3_part2","quote_type":"dialogue","speaker":"母亲","confidence":"high","reason":"引号前有'母亲就开始担心了'"}]}"""


def _load_env():
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


def resolve_quotes(parsed_result: Dict) -> Dict:
    """对 story_parser 输出中所有 quoted part 做语义判断，逐段落调用 LLM。

    Args:
        parsed_result: story_parser.parse_text() 的返回值

    Returns:
        包含每个 quoted part 判断结果的字典
    """
    env = _load_env()
    base_url = env.get("LLM_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
    api_key = env.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", ""))
    model = env.get("LLM_MODEL", os.environ.get("LLM_MODEL", "qwen3.6-plus"))

    if not base_url or not api_key:
        raise ValueError("缺少 LLM 配置。请在 .env 文件中设置 LLM_BASE_URL 和 LLM_API_KEY。")

    content_map = _build_content_map(parsed_result)
    all_resolutions = []
    total = len(parsed_result["paragraphs"])

    for para in parsed_result["paragraphs"]:
        pid = para["paragraph_id"]
        if not para["has_quotes"]:
            print(f"[{pid}/{total}] 跳过（无引号）")
            continue

        quoted_parts = [
            {
                "paragraph_id": para["paragraph_id"],
                "part_id": part["part_id"],
                "content": part["content"],
            }
            for part in para["parts"]
            if part["type"] == "quoted"
        ]

        if not quoted_parts:
            continue

        user_prompt = f"## 段落原文\n\n{para['text']}\n\n## 需要判断的引号\n\n"
        user_prompt += json.dumps(quoted_parts, ensure_ascii=False, indent=2)
        user_prompt += "\n\n请对以上每个引号判断 quote_type、speaker、confidence 和 reason。严格输出 JSON。"

        print(f"[{pid}/{total}] 处理中（{len(quoted_parts)} 个引号）...", end=" ", flush=True)
        raw_response = _call_llm(_SYSTEM_PROMPT, user_prompt, base_url, api_key, model)
        resolutions = _parse_response(raw_response)
        all_resolutions.extend(resolutions)
        print(f"完成")

    grouped = _group_by_paragraph(parsed_result, all_resolutions, content_map)

    return {
        "resolutions": grouped,
        "model": model,
        "total_resolved": len(all_resolutions),
    }


def _build_content_map(parsed_result: Dict) -> Dict[str, str]:
    """建立 part_id → content 的映射表。"""
    content_map = {}
    for para in parsed_result["paragraphs"]:
        for part in para["parts"]:
            content_map[part["part_id"]] = part["content"]
    return content_map


def _call_llm(
    system_prompt: str, user_prompt: str, base_url: str, api_key: str, model: str
) -> str:
    """调用 LLM API（Anthropic Messages 格式）。"""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/messages"
    else:
        url = f"{base}/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    response = requests.post(
        url, headers=headers, json=payload, timeout=(10, 120), verify=False
    )
    response.raise_for_status()

    data = response.json()
    for block in data["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise ValueError(f"LLM 响应中没有 text block: {data}")


def _parse_response(raw_response: str) -> List[Dict]:
    """解析 LLM 返回的 JSON。"""
    text = raw_response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    parsed = json.loads(text.strip())

    if "resolutions" in parsed:
        return parsed["resolutions"]

    return parsed if isinstance(parsed, list) else []


def _group_by_paragraph(
    parsed_result: Dict, resolutions: List[Dict], content_map: Dict[str, str]
) -> List[Dict]:
    """将 resolutions 按 paragraph_id 分组，关联原文信息。"""
    para_map = {p["paragraph_id"]: p for p in parsed_result["paragraphs"]}

    groups: Dict[int, List] = {}
    for r in resolutions:
        pid = r.get("paragraph_id")
        if pid is None:
            pid = int(r["part_id"].split("_")[0][1:])
        r["content"] = content_map.get(r["part_id"], "")
        groups.setdefault(pid, []).append(r)

    result = []
    for pid in sorted(groups.keys()):
        result.append(
            {
                "paragraph_id": pid,
                "paragraph_text": para_map[pid]["text"] if pid in para_map else "",
                "quote_resolutions": groups[pid],
            }
        )

    return result
