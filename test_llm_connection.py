import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL")

# 推荐 .env 里 LLM_BASE_URL 写到 /apps/anthropic，不带 /v1
if base_url.endswith("/v1"):
    url = f"{base_url}/messages"
else:
    url = f"{base_url}/v1/messages"

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

payload = {
    "model": model,
    "max_tokens": 64,
    "system": "你是一个简洁的中文助手。",
    "messages": [
        {"role": "user", "content": "只回复 OK"}
    ],
    "temperature": 0,
}

print("POST", url)
print("model =", model)

try:
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(10, 60),
        verify=False,
    )
    print("status =", resp.status_code)
    print(resp.text[:1000])
    resp.raise_for_status()
except requests.exceptions.ConnectTimeout:
    print("连接超时：大概率是网络/代理/域名访问问题。")
except requests.exceptions.ReadTimeout:
    print("读取超时：服务端响应慢，可能是模型慢/限流/请求排队。")
except requests.exceptions.ProxyError as e:
    print("代理错误：", e)
except requests.exceptions.SSLError as e:
    print("SSL 证书错误：", e)
except requests.exceptions.HTTPError as e:
    print("HTTP 错误：", e)
except Exception as e:
    print("其他错误：", type(e).__name__, e)