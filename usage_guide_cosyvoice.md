# Fun-CosyVoice3-0.5B 使用指南

## 服务信息

- **服务地址**: `http://10.50.121.102:8005`
- **模型**: Fun-CosyVoice3-0.5B-2512
- **功能**: 零样本语音克隆 + 自然语言指令控制

---

## 服务管理

### 启动服务

```bash
source /home/audiotest/miniconda3/bin/activate cosyvoice3
nohup python servers/api_server_cosyvoice.py --port 8005 --device cuda:6 > logs/cosyvoice.log 2>&1 &
```

### 停止服务

```bash
pkill -f api_server_cosyvoice
```

---

## API 接口

### 1. 健康检查

```bash
curl --noproxy '*' http://10.50.121.102:8005/health
```

**响应示例**:
```json
{"status":"ok","model":"Fun-CosyVoice3-0.5B"}
```

### 2. 配置信息

```bash
curl --noproxy '*' http://10.50.121.102:8005/v1/cosyvoice/config
```

**响应示例**:
```json
{"sampling_rate":24000,"supported_modes":["zero_shot","cross_lingual","instruct"]}
```

### 3. 语音生成

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8005/v1/cosyvoice/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"文本","prompt_text":"指令.<|endofprompt|>参考音频转写","prompt_audio":"base64编码的音频","mode":"zero_shot","stream":false}' \
  -o output.wav
```

---

## 参数说明

### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | **是** | - | 要合成的目标文本 |
| `prompt_text` | string | **是** | - | 指令 + 分隔符 + 参考音频转写 |
| `prompt_audio` | string | **是** | - | 参考音频（base64编码或文件路径） |
| `mode` | string | 否 | zero_shot | 生成模式 |
| `stream` | bool | 否 | false | 是否流式输出 |

### mode 生成模式

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `zero_shot` | 零样本克隆 | 保留参考音频的音色和语调 |
| `cross_lingual` | 跨语言+细粒度控制 | 控制呼吸[breath]、音素发音 |
| `instruct` | 指令控制 | 控制方言、语速、情绪 |

### prompt_text 格式（三种模式）

**zero_shot 模式：**
```
prompt_text = "指令.<|endofprompt|>参考音频转写"
```
| 部分 | 说明 |
|------|------|
| 指令部分 | 可选，控制输出语调（如"You are helpful"） |
| `<|endofprompt|>` | 分隔符，必填 |
| 参考音频转写 | 参考音频里实际说的内容 |

**instruct 模式：**
```
prompt_text = "指令.<|endofprompt|>"
```
| 部分 | 说明 |
|------|------|
| 指令部分 | 控制输出风格（方言、语速、情绪等），如"请用广东话表达" |
| `<|endofprompt|>` | 分隔符，必填 |

**cross_lingual 模式：**
```
text = "指令.<|endofprompt|>带控制符的文本"  # 控制符在text中
prompt_text = ""（空字符串）
```
控制符（如[breath]）放在text中，prompt_text留空。

---

## 指令控制示例 (instruct 模式)

| 效果 | prompt_text 示例 |
|------|------------------|
| 广东话 | `请用广东话表达。<\|endofprompt\|>` |
| 快速语速 | `请用尽可能快地语速说一句话。<\|endofprompt\|>` |
| 慢速 | `请用缓慢的语速说。<\|endofprompt\|>` |
| 高兴 | `用开心的语气说。<\|endofprompt\|>` |
| 生气 | `用生气的语气说。<\|endofprompt\|>` |
| 悲伤 | `用悲伤的语气说。<\|endofprompt\|>` |

---

## 细粒度控制示例 (cross_lingual 模式)

**cross_lingual 的 text 格式**：`指令.<|endofprompt|>带控制符的文本`

| 控制 | 用法 | 说明 |
|------|------|------|
| 呼吸 | `[breath]` | 在文本中插入呼吸声 |
| 中文拼音 | `[拼音]` | 控制中文读音 |
| 英文音素 | `[cmutn]` | 控制英文发音 |

```python
# 呼吸控制示例（cross_lingual 模式）
text = "You are helpful.<|endofprompt|>[breath]因为你那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络"
prompt_text = ""  # cross_lingual 不需要
mode = "cross_lingual"
```


---

## 支持语言和方言

### 主要语言 (9种)

| 语言 | 参数值 |
|------|--------|
| 中文 | Chinese |
| 英文 | English |
| 日语 | Japanese |
| 韩语 | Korean |
| 德语 | German |
| 西班牙语 | Spanish |
| 法语 | French |
| 意大利语 | Italian |
| 俄语 | Russian |

### 中文方言 (18+种)

广东话、闽南语、四川话、东北话、上海话、天津话、山东话、山西话、陕西话、宁夏话、甘肃话等

---

## Python 客户端调用

### 代码示例

```python
import requests
import base64

url = "http://10.50.121.102:8005/v1/cosyvoice/generate"

# 读取参考音频并转为base64
with open("prompt.wav", 'rb') as f:
    prompt_audio = base64.b64encode(f.read()).decode('utf-8')

data = {
    "text": "你到底有没有在听我说话？说了这么多遍还是记不住！",
    "prompt_text": "请用生气的语气说。<|endofprompt|>",  # instruct模式：指令+分隔符
    "prompt_audio": prompt_audio,
    "mode": "instruct",
    "stream": False
}

response = requests.post(url, json=data, proxies={"http": None, "https": None})

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("音频已生成: output.wav")
```

---

## curl 调用示例

### 基础零样本克隆

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8005/v1/cosyvoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "要合成的目标文本",
    "prompt_text": "You are helpful.<|endofprompt|>希望你以后能够做的比我还好哟！",
    "prompt_audio": "/path/to/prompt.wav",
    "mode": "zero_shot",
    "stream": false
  }' \
  -o output.wav
```

### 指令控制（生气语气）

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8005/v1/cosyvoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "说了多少遍了还记不住，真是让人火大！",
    "prompt_text": "用生气的语气说。<|endofprompt|>",  # instruct模式：指令+分隔符
    "prompt_audio": "/path/to/prompt.wav",
    "mode": "instruct",
    "stream": false
  }' \
  -o output.wav
```

### 呼吸控制（cross_lingual 模式）

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8005/v1/cosyvoice/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "You are helpful.<|endofprompt|>[breath]因为你那一辈人[breath]在乡里面住要习惯一点",  # text中直接用控制符
    "prompt_text": "",
    "prompt_audio": "/path/to/prompt.wav",
    "mode": "cross_lingual",
    "stream": false
  }' \
  -o output.wav
```

---

## 注意事项

1. **--noproxy 参数**: 使用 `curl` 时必须加 `--noproxy '*'`
2. **Python proxies**: 使用 `requests` 时设置 `proxies={"http": None, "https": None}`
3. **prompt_audio 格式**: 可以是文件路径（如 `/path/to/prompt.wav`）或 base64 编码
4. **`<|endofprompt|>`**: 必须有分隔符
5. **参考音频**: 建议5-30秒，太短效果差，太长处理慢
6. **三种模式的 prompt_text 格式**：
   - `zero_shot`: `"指令.<|endofprompt|>转写"`（包含转写）
   - `instruct`: `"指令.<|endofprompt|>"`（只有指令）
   - `cross_lingual`: `""`（空字符串，text中直接用控制符+分隔符）