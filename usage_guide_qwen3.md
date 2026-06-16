# Qwen3-TTS VoiceDesign 使用指南

## 服务信息

- **服务地址**: `http://10.154.39.97:8007`
- **模型**: Qwen3-TTS-12Hz-1.7B-VoiceDesign
- **功能**: 通过自然语言描述生成指定音色的语音

---

## 服务管理

### 启动服务

```bash
source /home/audiotest/miniconda3/bin/activate qwen3-tts
nohup python servers/api_server_qwen3_voicedesign.py --port 8007 --device cuda:4 > logs/qwen3.log 2>&1 &
```

### 停止服务

```bash
pkill -f api_server_qwen3_voicedesign
```

---

## API 接口

### 1. 健康检查

```bash
curl --noproxy '*' http://10.154.39.97:8007/health
```

**响应示例**:
```json
{"status":"ok","model":"Qwen3-TTS-VoiceDesign"}
```

### 2. 配置信息

```bash
curl --noproxy '*' http://10.154.39.97:8007/v1/voicedesign/config
```

### 3. 语音生成

```bash
curl --noproxy '*' -X POST http://10.154.39.97:8007/v1/voicedesign/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "要合成的文本", "instruction": "声音风格描述", "language": "Chinese"}' \
  -o output.wav
```

---

## 参数说明

### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | **是** | - | 要合成的文本，支持中英文 |
| `instruction` | string | **是** | - | 声音风格描述，可用自然语言指定音色、情绪、语速等 |
| `language` | string | **是** | - | 语言，支持：Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| `max_new_tokens` | int | 否 | 2048 | 最大生成token数，控制输出长度 |

### instruction 风格描述示例

| 风格 | instruction 示例 |
|------|-----------------|
| 活泼女声 | "活泼开心的女声，语气轻快，带着愉悦的笑容感" |
| 磁性男声 | "低沉磁性的男性声音，语速缓慢，成熟稳重" |
| 萝莉音 | "稚嫩清脆的小女孩声音，天真可爱" |
| 老年声音 | "苍老沙哑的老年声音，语速偏慢" |
| 新闻播报 | "专业播音员的声音，吐字清晰，语调平稳" |
| 愤怒 | "愤怒的语气，情绪激动，声音提高" |
| 悲伤 | "伤感的语气，情绪低落，语速缓慢" |

---

## Python 客户端调用

### 安装依赖

```bash
pip install requests soundfile
```

### 代码示例

```python
import requests

url = "http://10.154.39.97:8007/v1/voicedesign/generate"

data = {
    "text": "今天天气真不错呀！阳光明媚心情好，去公园散步吧！",
    "instruction": "活泼开心的女声，语气轻快，带着愉悦的笑容感",
    "language": "Chinese"
}

response = requests.post(url, json=data, proxies={"http": None, "https": None})

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("音频已生成: output.wav")
else:
    print(f"请求失败: {response.status_code}")
```

---

## curl 调用示例

### 基础调用

```bash
curl --noproxy '*' -X POST http://10.154.39.97:8007/v1/voicedesign/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，欢迎使用Qwen3-TTS语音合成",
    "instruction": "温柔的成年女性声音，语气平和亲切",
    "language": "Chinese"
  }' \
  -o output.wav
```

### 英文语音

```bash
curl --noproxy '*' -X POST http://10.154.39.97:8007/v1/voicedesign/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a voice generation test",
    "instruction": "Friendly American male voice, casual and warm",
    "language": "English"
  }' \
  -o output.wav
```

### 控制输出长度

```bash
curl --noproxy '*' -X POST http://10.154.39.97:8007/v1/voicedesign/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "短文本",
    "instruction": "活泼女声",
    "language": "Chinese",
    "max_new_tokens": 512
  }' \
  -o output.wav
```

---

## 支持的语言

| 语言 | language 参数值 |
|------|----------------|
| 中文 | Chinese |
| 英文 | English |
| 日语 | Japanese |
| 韩语 | Korean |
| 德语 | German |
| 法语 | French |
| 俄语 | Russian |
| 葡萄牙语 | Portuguese |
| 西班牙语 | Spanish |
| 意大利语 | Italian |

---

## 注意事项

1. **--noproxy 参数**: 使用 `curl` 时必须加 `--noproxy '*'`，否则会被代理拦截
2. **Python proxies**: 使用 `requests` 时设置 `proxies={"http": None, "https": None}` 禁用代理
3. **instruction 描述**: 越详细的描述生成效果越好，包括：性别、年龄、情绪、语速、语气等
4. **服务端口**: 默认 8007，确保端口未被占用