# IndexTTS-2 使用指南

## 概述

IndexTTS-2 是新一代情感表达和时长可控的零样本 TTS 模型（Bilibili 出品），支持：
- 零样本语音克隆
- 情感控制（与音色解耦，独立控制）
- 时长控制
- 拼音控制
- 多模态情感输入（向量/音频/文本）

---

## 服务信息

| 项目 | 值 |
|------|-----|
| **服务地址** | `http://10.154.39.97:8009` |
| **模型** | IndexTTS-2 |
| **采样率** | 22050 Hz |
| **显存需求** | ~12GB |

---

## 启动服务

```bash
source ~/miniconda3/bin/activate indextts
export LD_LIBRARY_PATH=$HOME/cuda-12.9/lib64:$LD_LIBRARY_PATH
nohup python servers/api_server_indextts_8009.py --port 8009 --device cuda:6 > logs/indextts_8009.log 2>&1 &
```

停止服务：
```bash
pkill -f api_server_indextts_8009
```

---

## API 接口

### 1. 健康检查

```bash
curl http://10.154.39.97:8009/health
```

**响应示例**:
```json
{"status":"ok","model":"IndexTTS 2 (Full)","device":"cuda:6","sampling_rate":22050}
```

### 2. 模型配置

```bash
curl http://10.154.39.97:8009/v1/tts/config
```

### 3. 语音合成

```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"要合成的文本"}' \
  -o output.wav
```

---

## 参数说明

### 核心参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | **是** | - | 要合成的文本，支持中英文混合 |
| `reference_audio_base64` | string | 否 | 内置默认 | 音色参考音频 base64 编码数据（与 reference_audio_path 二选一） |
| `reference_audio_path` | string | 否 | 内置默认 | 音色参考音频路径（服务器本地路径，用于音色克隆） |
| `emotion_audio_base64` | string | 否 | null | 情感参考音频 base64 编码数据（与 emotion_audio_path 二选一） |
| `emotion_audio_path` | string | 否 | null | 情感参考音频路径（与音色分离） |
| `emotion_alpha` | float | 否 | 1.0 | 情感强度 0.0-1.0，1.0=100%情感 |
| `emotion_vector` | list | 否 | null | 8维情绪向量 |
| `emotion_text` | string | 否 | null | 情绪文本描述（自动生成情绪向量） |
| `use_random` | bool | 否 | false | 是否启用随机性（会降低克隆保真度） |

### 采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.8 | 采样温度 0-1，越高越随机 |
| `top_p` | float | 0.8 | Nucleus采样 cutoff 0-1 |
| `top_k` | int | 30 | Top-k 采样 |
| `num_beams` | int | 3 | Beam search 数量，越高质量越好但更慢 |
| `repetition_penalty` | float | 10.0 | 重复惩罚 |

### 长度控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_text_tokens` | int | 120 | 每段最大文本token数 |
| `max_mel_tokens` | int | 1500 | 最大生成mel token数，控制生成时长 |
| `interval_silence` | int | 200 | 音频片段间的静音时长(毫秒) |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verbose` | bool | false | 是否输出详细信息 |
| `output_path` | string | null | 输出路径（可选） |

---

## 功能用法

### 1. 基础音色克隆

使用参考音频克隆音色（推荐使用 base64 编码，无需服务器本地文件）：

```python
import base64
import requests

with open("/path/to/voice.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "你好，欢迎使用IndexTTS-2语音合成系统。",
        "reference_audio_base64": audio_b64
    },
    proxies={"http": None, "https": None},
    proxies={"http": None, "https": None},
)

with open("output.wav", "wb") as f:
    f.write(response.content)
print("音频已生成: output.wav")
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，欢迎使用IndexTTS-2语音合成系统。","reference_audio_base64":"<base64字符串>"}' \
  -o output.wav
```

> **推荐**：使用 `reference_audio_base64` 传递音频数据，无需服务器本地文件。
> 如使用 `reference_audio_path`，需确保服务器本地可访问该路径。

---

### 2. 情感控制 - 使用情绪向量

使用8维情绪向量独立控制情感：

| 维度 | 情绪 | 说明 |
|------|------|------|
| [0] | happy | 开心、愉快 |
| [1] | angry | 愤怒、生气 |
| [2] | sad | 悲伤、难过 |
| [3] | afraid | 恐惧、害怕 |
| [4] | disgusted | 厌恶、反感 |
| [5] | melancholic | 忧郁、哀伤 |
| [6] | surprised | 惊讶、吃惊 |
| [7] | calm | 平静、冷静 |

```python
# 悲伤的情感
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "对不起嘛！我的记性真的不太好，和你在一起的事情，我都会努力记住的。",
        "reference_audio_base64": "<base64_音频数据>",
        "emotion_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
        "use_random": False
    }
)

# 惊恐的情感
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "快躲起来！是他要来了！他要来抓我们了！",
        "reference_audio_base64": "<base64_音频数据>",
        "emotion_vector": [0, 0, 0, 0.8, 0, 0, 0, 0],
        "emo_alpha": 0.6
    }
)

# 开心的情感
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "今天天气真不错，心情特别愉快！",
        "reference_audio_base64": "<base64_音频数据>",
        "emotion_vector": [0.8, 0, 0, 0, 0, 0, 0, 0.2]
    }
)
```

使用 curl：
```bash
# 悲伤
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"这个消息太让人难过了。","reference_audio_base64":"<base64字符串>","emotion_vector":[0,0,0.8,0,0,0,0,0]}' \
  -o sad.wav

# 开心
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气真不错！","reference_audio_base64":"<base64字符串>","emotion_vector":[0.8,0,0,0,0,0,0,0.2]}' \
  -o happy.wav
```

---

### 3. 情感控制 - 使用情感参考音频

使用独立的情感参考音频，与音色分离控制：

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "酒楼丧尽天良，开始借机竞拍房间。",
        "reference_audio_base64": "<base64_音色音频>",       # 音色参考
        "emotion_audio_base64": "<base64_情感音频>",       # 情感参考（独立控制）
        "emotion_alpha": 0.9
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"酒楼丧尽天良，开始借机竞拍房间。","reference_audio_base64":"<base64_音色音频>","emotion_audio_base64":"<base64_情感音频>","emotion_alpha":0.9}' \
  -o emotion_clone.wav
```

**说明**：
- `reference_audio_base64` / `reference_audio_path`: 控制音色（声音像谁）
- `emotion_audio_base64` / `emotion_audio_path`: 控制情感（情绪表达方式）
- 两者可以完全不同，实现音色和情感的独立控制

---

### 4. 情感控制 - 使用情绪文本描述

通过文本描述自动生成情绪向量：

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "快躲起来！是他要来了！",
        "reference_audio_base64": "<base64_音频数据>",
        "emotion_text": "你吓死我了！你是鬼吗？",
        "emo_alpha": 0.6
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"快躲起来！是他要来了！","reference_audio_base64":"<base64字符串>","emotion_text":"你吓死我了！你是鬼吗？","emo_alpha":0.6}' \
  -o text_emotion.wav
```

> **建议**：使用 `emo_alpha` 约 0.6 或更低，可获得更自然的语音效果。

---

### 5. 拼音控制

在文本中使用拼音标注精确控制发音：

```
之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，
如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
```

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2。",
        "reference_audio_base64": "<base64_音频数据>"
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2。","reference_audio_base64":"<base64字符串>"}' \
  -o pinyin.wav
```

> **注意**：拼音控制只支持有效的汉语拼音组合，完整列表见 `models/tts/IndexTTS-2/pinyin.vocab`。

---

### 6. 随机性控制

使用 `use_random=True` 引入随机性，会降低克隆保真度但增加变化：

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "这是一个带有随机变化的语音合成。",
        "reference_audio_base64": "<base64_音频数据>",
        "use_random": True,
        "emotion_vector": [0.5, 0, 0, 0, 0, 0, 0.3, 0.2]
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"这是一个带有随机变化的语音合成。","reference_audio_base64":"<base64字符串>","use_random":true,"emotion_vector":[0.5,0,0,0,0,0,0.3,0.2]}' \
  -o random.wav
```

---

### 7. 采样参数调优

调整采样参数控制生成质量：

```python
# 高质量（更慢）
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "高质量语音合成示例。",
        "reference_audio_base64": "<base64_音频数据>",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "num_beams": 5,
        "repetition_penalty": 8.0
    }
)

# 快速（质量略低）
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "快速语音合成示例。",
        "reference_audio_base64": "<base64_音频数据>",
        "temperature": 0.9,
        "top_p": 0.7,
        "top_k": 20,
        "num_beams": 1,
        "repetition_penalty": 12.0
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"高质量语音合成示例。","reference_audio_base64":"<base64字符串>","temperature":0.7,"top_p":0.9,"top_k":50,"num_beams":5,"repetition_penalty":8.0}' \
  -o high_quality.wav
```

**参数说明**：
- `temperature`: 降低更稳定，提高更多样
- `top_p`: 提高增加多样性
- `top_k`: 提高扩大采样范围
- `num_beams`: 增加提高质量（但更慢）
- `repetition_penalty`: 控制重复发音

---

### 8. 时长控制

通过 `max_mel_tokens` 和 `max_text_tokens` 控制生成时长：

```python
# 短语音
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "短句。",
        "reference_audio_base64": "<base64_音频数据>",
        "max_mel_tokens": 500
    }
)

# 长语音
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "这是一段很长的语音内容。",
        "reference_audio_base64": "<base64_音频数据>",
        "max_mel_tokens": 3000,
        "max_text_tokens": 100
    }
)
```

使用 curl：
```bash
# 短语音
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"短句。","reference_audio_base64":"<base64字符串>","max_mel_tokens":500}' \
  -o short.wav

# 长语音
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"这是一段很长的语音内容。","reference_audio_base64":"<base64字符串>","max_mel_tokens":3000,"max_text_tokens":100}' \
  -o long.wav
```

---

### 9. 静音间隔控制

通过 `interval_silence` 控制音频片段间的静音时长（毫秒）：

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "第一段文本。第二段文本。第三段文本。",
        "reference_audio_base64": "<base64_音频数据>",
        "interval_silence": 500  # 500毫秒静音间隔
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"第一段文本。第二段文本。","reference_audio_base64":"<base64字符串>","interval_silence":500}' \
  -o with_pause.wav
```

---

### 10. 英文合成

```python
response = requests.post(
    "http://10.154.39.97:8009/v1/tts/synthesize",
    json={
        "text": "Hello, this is a voice synthesis test using IndexTTS2.",
        "reference_audio_base64": "<base64_音频数据>"
    }
)
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.154.39.97:8009/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a voice synthesis test using IndexTTS2.","reference_audio_base64":"<base64字符串>"}' \
  -o english.wav
```

---

## 情绪向量参考表

| 情绪 | emotion_vector 示例 | 场景 |
|------|---------------------|------|
| 开心 | `[0.8, 0, 0, 0, 0, 0, 0, 0.2]` | 高兴、愉悦 |
| 生气 | `[0, 0.8, 0, 0, 0, 0, 0, 0.2]` | 愤怒、烦躁 |
| 悲伤 | `[0, 0, 0.8, 0, 0, 0.1, 0, 0.1]` | 难过、沮丧 |
| 恐惧 | `[0, 0, 0, 0.8, 0, 0, 0, 0.2]` | 害怕、惊恐 |
| 厌恶 | `[0, 0, 0, 0, 0.8, 0, 0, 0.2]` | 厌恶、反感 |
| 忧郁 | `[0, 0, 0, 0, 0, 0.8, 0, 0.2]` | 忧郁、哀伤 |
| 惊讶 | `[0, 0, 0, 0, 0, 0, 0.8, 0.2]` | 惊讶、震惊 |
| 平静 | `[0, 0, 0, 0, 0, 0, 0, 1.0]` | 平静、自然 |

> **注意**：情绪向量总和不应超过 0.8，模型会自动缩放超出的值。

---

## 内置参考音频

默认使用内置参考音频。不传 `reference_audio_base64` 或 `reference_audio_path` 时使用此音频作为音色参考。

---

## 注意事项

1. **参考音频质量**：建议 5-15 秒清晰无噪声的音频
2. **文本长度**：长文本会自动分段，每段不超过 `max_text_tokens` 个token
3. **情感强度**：`emo_alpha` 建议 0.6-1.0，过高可能发音不清晰
4. **拼音标注**：只支持有效的汉语拼音组合
5. **随机性**：`use_random=True` 会降低克隆保真度
6. **输出采样率**：22050 Hz

---

## 与其他 TTS 模型对比

| 特性 | IndexTTS-2 | CosyVoice | Qwen3-TTS |
|------|------------|-----------|-----------|
| 采样率 | 22050 Hz | 24000 Hz | 24000 Hz |
| 情感控制 | 8维向量 + 参考音频 + 文本 | instruct 模式指令 | 自然语言指令 |
| 情感解耦 | 支持（音色+情感独立控制） | 不支持 | 不支持 |
| 细粒度控制 | 拼音控制 | [breath] 等控制符 | 自然语言描述 |
| 语言支持 | 中英文为主 | 9种语言 + 18种方言 | 10种语言 |
| 时长控制 | 支持（精确 + 自适应） | 不支持 | 支持 |
| 音色克隆 | 零样本 | 零样本 | 设计生成 |