# MOSS-VoiceGenerator 使用指南

## 服务信息

- **服务地址**: `http://10.50.121.102:8003`
- **模型**: MOSS-VoiceGenerator (OpenMOSS)
- **采样率**: 24000 Hz
- **功能**: 通过自然语言文本描述生成指定音色的语音（无需参考音频）

---

## 模型特点

| 特点 | 说明 |
|------|------|
| **零样本语音设计** | 通过文本描述直接生成指定音色，无需参考音频 |
| **高情感表达** | 生成具有动态情感表现的语音 |
| **类人自然度** | 真实的呼吸、停顿和声音细节 |
| **多语言支持** | 高质量的中文和英文合成 |
| **风格控制** | 支持情感、语速、音高、声音特征等精细控制 |

---

## 模型架构

MOSS-VoiceGenerator 采用 **MossTTSDelay** 架构，将语音描述指令和待合成文本拼接后联合 tokenize 输入，实现：
- 音色设计（Timbre Design）
- 风格控制（Style Control）
- 内容合成（Content Synthesis）

---

## 服务管理

### 启动服务

```bash
source ~/miniconda3/bin/activate moss-tts
nohup python servers/api_server_voicegen.py --port 8003 --device cuda:4 > logs/voicegen.log 2>&1 &
```

### 停止服务

```bash
pkill -f api_server_voicegen
```

---

## API 接口

### 1. 健康检查

```bash
curl --noproxy '*' http://10.50.121.102:8003/health
```

**响应示例**:
```json
{"status":"ok","model":"MOSS-VoiceGenerator"}
```

### 2. 配置信息

```bash
curl --noproxy '*' http://10.50.121.102:8003/v1/voicegen/config
```

**响应示例**:
```json
{"sampling_rate":24000,"supported_features":["voice_generation","instruction_control"]}
```

### 3. 语音生成

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"要合成的文本","instruction":"声音风格描述"}' \
  -o output.wav
```

---

## 参数说明

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | **是** | 要合成的文本，支持中文和英文 |
| `instruction` | string | **是** | 声音风格描述（如：情感、语速、音高、声音特征） |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audio_temperature` | 1.5 | 温度参数，较高值增加变化，较低值稳定韵律 |
| `audio_top_p` | 0.6 | Nucleus 采样截断，较低值更保守 |
| `audio_top_k` | 50 | Top-K 采样，较低值缩小采样空间 |
| `audio_repetition_penalty` | 1.1 | >1.0 抑制重复模式 |

> **注意**: MOSS-VoiceGenerator 对采样参数敏感，建议使用默认参数或仅微调。

---

## 调用示例

### 1. 老年声音

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "哎呀，我的老腰啊，这年纪大了就是不行了。",
    "instruction": "疲惫沙哑的老年声音缓慢抱怨，带有轻微呻吟。"
  }' \
  -o elderly.wav
```

### 2. 美食主持人

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "亲爱的观众们，今天我要为大家做一道传说中的龙须面，这道面条细如发丝，需要极其精湛的手艺才能制作成功。",
    "instruction": "热情的美食节目主持人，语调生动活泼，充满对美食的热爱和专业精神。"
  }' \
  -o host.wav
```

### 3. 酒吧老板（英文）

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hey there, stranger! What brings you to our humble town? Looking for a good drink or a tall tale?",
    "instruction": "Hearty, jovial tavern owner'\''s voice, loud and welcoming with a slightly gruff, friendly tone in American English."
  }' \
  -o bartender.wav
```

### 4. 中性发音练习（英文）

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "instruction": "Clear, neutral voice for phonetic practice, even tempo and precise articulation in standard American English."
  }' \
  -o neutral.wav
```

### 5. 开心语气

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "太棒了！我们的项目终于完成了！大家辛苦了！",
    "instruction": "开心的语气，语速轻快，充满喜悦和兴奋。"
  }' \
  -o happy.wav
```

### 6. 悲伤语气

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "一切都结束了，我们再也回不去了。",
    "instruction": "悲伤低沉的语调，语速缓慢，带着无奈和失落。"
  }' \
  -o sad.wav
```

### 7. 生气语气

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你怎么又做错了！说了多少遍都记不住！",
    "instruction": "愤怒激动的语气，声音提高，语速加快，带有不耐烦。"
  }' \
  -o angry.wav
```

### 8. 播音员

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8003/v1/voicegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "各位观众晚上好，这里是新闻联播节目。",
    "instruction": "专业播音员的声音，吐字清晰，语调平稳庄重。"
  }' \
  -o broadcaster.wav
```

---

## Python 客户端调用

### 基础用法

```python
import requests

url = "http://10.50.121.102:8003/v1/voicegen/generate"

data = {
    "text": "欢迎使用 MOSS-VoiceGenerator 语音合成系统。",
    "instruction": "温柔亲切的女声，语速适中，自然流畅。"
}

response = requests.post(url, json=data, proxies={"http": None, "https": None}, timeout=300)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("音频已生成: output.wav")
else:
    print(f"请求失败: {response.status_code}")
```

### 使用 VoiceGenClient 类

```python
import sys
sys.path.append("/data3/c00657214/tts_project")
from clients.voicegen_client import VoiceGenClient

client = VoiceGenClient(base_url="http://10.50.121.102:8003")

# 检查服务状态
if not client.health_check():
    print("服务未启动")
    exit(1)

# 获取配置
config = client.get_config()
print(f"配置: {config}")

# 生成音频
audio = client.generate(
    text="今天天气真不错！",
    instruction="活泼开心的女声，语气轻快愉悦",
    output_path="output.wav"
)
print(f"生成完成: {audio.shape}")
```

---

## instruction 风格描述参考

| 风格 | instruction 示例 |
|------|-----------------|
| 老年声音 | "疲惫沙哑的老年声音缓慢抱怨，带有轻微呻吟。" |
| 美食主持人 | "热情的美食节目主持人，语调生动活泼。" |
| 酒吧老板 | "豪爽热情的酒吧老板，声音洪亮热情。" |
| 播音员 | "专业播音员，吐字清晰，语调平稳。" |
| 萝莉音 | "稚嫩清脆的小女孩声音，天真可爱。" |
| 磁性男声 | "低沉磁性的男性声音，成熟稳重。" |
| 开心 | "开心的语气，语速轻快，充满喜悦。" |
| 悲伤 | "悲伤低沉的语调，语速缓慢。" |
| 生气 | "愤怒激动的语气，声音提高。" |
| 紧张 | "紧张害怕的语气，微微颤抖。" |

---

## 注意事项

1. **--noproxy 参数**: 使用 `curl` 时必须加 `--noproxy '*'`
2. **Python proxies**: 使用 `requests` 时设置 `proxies={"http": None, "https": None}`
3. **采样参数**: 模型对采样参数敏感，建议使用默认参数
4. **指令描述**: instruction 越详细，生成效果越好
5. **超时**: 建议设置 300 秒以上的超时时间

---

## 与其他 TTS 模型对比

| 特性 | MOSS-VoiceGenerator | CosyVoice | IndexTTS-2 | Qwen3-TTS |
|------|---------------------|-----------|------------|-----------|
| 采样率 | 24000 Hz | 24000 Hz | 22050 Hz | 24000 Hz |
| 音色获取 | 文本描述生成 | 需要参考音频 | 需要参考音频 | 文本描述生成 |
| 情感控制 | 指令描述 | instruct 模式 | 8维向量 + 文本 | 自然语言指令 |
| 零样本克隆 | 不需要参考音频 | 需要参考音频 | 需要参考音频 | 不需要参考音频 |
| 多语言 | 中英文 | 9种语言 + 方言 | 中英文 | 10种语言 |

---

## 推荐采样参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `audio_temperature` | 1.5 | 平衡多样性和稳定性 |
| `audio_top_p` | 0.6 | 保守采样 |
| `audio_top_k` | 50 | 适度限制 |
| `audio_repetition_penalty` | 1.1 | 防止重复 |