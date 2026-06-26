# Fish Audio S2-Pro 使用指南

## 概述

Fish Audio S2-Pro 是领先的文本转语音（TTS）模型，具有精细的韵律和情感内联控制能力。基于超过 1000 万小时、80+ 语言音频数据训练，结合强化学习对齐和双自回归（Dual-AR）架构。

**核心特性**：
- 精细内联控制（通过 `[tag]` 语法）
- 多说话人对话支持
- 80+ 语言支持
- 15000+ 独特标签 + 自由文本描述
- 高保真 44100 Hz 输出

---

## 服务信息

| 项目 | 值 |
|------|-----|
| **服务地址** | `http://10.50.121.102:8006` |
| **模型** | Fish Audio S2-Pro |
| **采样率** | 44100 Hz |
| **架构** | Dual-Autoregressive (Dual-AR) |
| **显存需求** | ~11GB+ |
| **音色克隆** | **支持**（需 `reference_audio` + `prompt_text`；详见第 15 节） |
| **多说话人** | 支持（`<\|speaker:N\|>` 标签，详见第 9 节） |
| **情绪控制** | 15000+ 内联标签 + 自由文本描述 |
| **本地 wrapper 局限** | 当前未暴露 `reference_audio` / `prosody` 等高级字段（详见第 18 节） |

---

## 启动服务

```bash
source ~/miniconda3/bin/activate s2-pro
export LD_LIBRARY_PATH=$HOME/cuda-12.9/lib64:$LD_LIBRARY_PATH
nohup python servers/api_server_s2pro.py --model-path ./s2-pro --host 0.0.0.0 --port 8006 --device cuda:0 > logs/s2pro.log 2>&1 &
```

停止服务：
```bash
pkill -f api_server_s2pro
```

---

## API 接口

### 1. 健康检查

```bash
curl --noproxy '*' http://10.50.121.102:8006/health
```

**响应示例**:
```json
{"status":"ok","model_loaded":true,"device":"cuda:0","sampling_rate":44100}
```

### 2. 模型配置

```bash
curl --noproxy '*' http://10.50.121.102:8006/v1/voicegen/config
```

**响应示例**:
```json
{"model":"s2-pro","architecture":"fish_qwen3_omni (Dual-AR)","sampling_rate":44100,"device":"cuda:0"}
```

### 3. 语音生成

```bash
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=要合成的文本" \
  -F "instruction=风格指令" \
  -o output.wav
```

---

## 参数说明

### 核心参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | **是** | - | 要合成的文本，支持中英文、标签语法 |
| `instruction` | string | 否 | "" | 全局风格指令（应用于整个音频） |
| `max_new_tokens` | int | 否 | 4096 | 最大生成token数（控制音频长度） |
| `temperature` | float | 否 | 1.0 | 采样温度 (0, 2)，越高越随机 |
| `top_p` | float | 否 | 0.6 | Nucleus采样 cutoff (0, 1) |

---

## 功能用法

### 1. 基础语音合成

```python
import requests

response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "你好，欢迎使用 Fish Audio S2-Pro 语音合成系统。"},
    proxies={"http": None, "https": None},
    timeout=300
)

with open("output.wav", "wb") as f:
    f.write(response.content)
print("音频已生成: output.wav")
```

使用 curl：
```bash
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=你好，欢迎使用 Fish Audio S2-Pro 语音合成系统。" \
  -o output.wav
```

---

### 2. 内联标签控制 - 情感标签

通过 `[tag]` 语法在文本中嵌入情感控制标签：

```python
# 兴奋语气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[excited]太棒了！我们成功了！"},
)

# 愤怒语气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[angry]你怎么可以这样对我！"},
)

# 悲伤语气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[sad]一切都结束了..."},
)

# 惊讶语气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[surprised]真的吗？"},
)

# 耳语
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[whisper]这是个秘密，只能告诉你一个人。"},
)

# 尖叫
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[screaming]救命啊！着火了！"},
)
```

使用 curl：
```bash
# 兴奋
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[excited]太棒了！我们成功了！" \
  -o excited.wav

# 愤怒
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[angry]你怎么可以这样对我！" \
  -o angry.wav

# 悲伤
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[sad]一切都结束了..." \
  -o sad.wav

# 耳语
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[whisper]这是个秘密。" \
  -o whisper.wav
```

---

### 3. 内联标签控制 - 韵律标签

控制停顿、强调等韵律特征：

```python
# 插入停顿
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "Hello [pause] World"},
)

# 短停顿
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "one [short pause] two [short pause] three"},
)

# 强调
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "这才是 [emphasis] 重点"},
)

# 叹气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[sigh] 唉，又失败了。"},
)

# 打断
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "等一下 [interrupting] 我有话说。"},
)
```

使用 curl：
```bash
# 停顿
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=Hello [pause] World" \
  -o pause.wav

# 短停顿
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=one [short pause] two [short pause] three" \
  -o short_pause.wav

# 强调
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=这才是 [emphasis] 重点" \
  -o emphasis.wav

# 叹气
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[sigh] 唉，又失败了。" \
  -o sigh.wav
```

---

### 4. 内联标签控制 - 声音效果标签

模拟各种声音效果：

```python
# 笑声
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[laughing] 哈哈，真有趣！"},
)

# 轻笑
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[chuckle] 有点意思。"},
)

# 吸气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[inhale] 呼...这味道真好闻。"},
)

# 呼气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[exhale] 终于完成了。"},
)

# 清嗓子
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[clearing throat] 嗯...我来说几句。"},
)

# 喘息
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[panting] 我...跑不动了..."},
)

# 抽鼻子
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[sniff] 好像要下雨了。"},
)
```

使用 curl：
```bash
# 笑声
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[laughing] 哈哈，真有趣！" \
  -o laughing.wav

# 轻笑
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[chuckle] 有点意思。" \
  -o chuckle.wav

# 吸气
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[inhale] 呼...这味道真好闻。" \
  -o inhale.wav
```

---

### 5. 内联标签控制 - 音量标签

控制音量大小：

```python
# 大声
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[loud] 注意听好了！"},
)

# 小声
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[quiet] 这里说话要小声点。"},
)

# 提高音量
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[volume up] 大家听好了！"},
)

# 降低音量
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[volume down] 这是个秘密..."},
)
```

使用 curl：
```bash
# 大声
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[loud] 注意听好了！" \
  -o loud.wav

# 小声
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[quiet] 这里说话要小声点。" \
  -o quiet.wav
```

---

### 6. 内联标签控制 - 特殊效果标签

```python
# 唱歌
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[singing] 哆来咪发梭拉西~"},
)

# 回声
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[echo] 回声效果测试..."},
)

# 带口音
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[with strong accent] This is a test."},
)

# 啧啧声
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[tsk] 不行不行。"},
)

# 呻吟
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[moaning] 唉..."},
)

# 愉悦
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[delight] 太完美了！"},
)
```

使用 curl：
```bash
# 唱歌
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[singing] 哆来咪发梭拉西~" \
  -o singing.wav

# 回声
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[echo] 回声效果测试..." \
  -o echo.wav

# 带口音
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[with strong accent] This is a test." \
  -o accent.wav
```

---

### 7. 自由文本标签控制

S2-Pro 支持**自由文本描述**而非固定标签：

```python
# 小声耳语
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[whisper in small voice] 我们要悄悄溜走。"},
)

# 专业播音语调
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[professional broadcast tone] 各位观众晚上好。"},
)

# 音调提高
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[pitch up] 这太不可思议了！"},
)

# 音调降低
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[pitch down] 这下麻烦了..."},
)

# 慢速说话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[speak slowly] 请大家...注意...安全..."},
)

# 快速说话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[speak quickly] 快快快来不及了！"},
)

# 悲伤疲惫
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[sad and tired] 我真的...好累..."},
)

# 开心兴奋
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[happy and excited] 我们赢了！太棒了！"},
)
```

使用 curl：
```bash
# 小声耳语
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[whisper in small voice] 我们要悄悄溜走。" \
  -o whisper_text.wav

# 专业播音
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[professional broadcast tone] 各位观众晚上好。" \
  -o broadcast.wav

# 慢速说话
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[speak slowly] 请大家...注意...安全..." \
  -o slow.wav
```

---

### 8. 全局指令控制 (instruction)

通过 `instruction` 参数设置整个音频的整体风格：

```python
# 整体兴奋语气
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "我们的项目终于完成了！大家一起庆祝吧！",
        "instruction": "[excited]"
    },
)

# 整体耳语风格
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "嘘...老板来了，快假装工作。",
        "instruction": "[whisper]"
    },
)

# 播音员风格
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "欢迎收看本期节目。",
        "instruction": "[professional broadcast tone]"
    },
)

# 中文描述
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "这是一段温柔亲切的语音。",
        "instruction": "温柔亲切的女声"
    },
)
```

使用 curl：
```bash
# 整体兴奋
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=我们的项目终于完成了！大家一起庆祝吧！" \
  -F "instruction=[excited]" \
  -o excited_global.wav

# 整体耳语
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=嘘...老板来了，快假装工作。" \
  -F "instruction=[whisper]" \
  -o whisper_global.wav
```

> **说明**：instruction 与内联标签的区别：
> - instruction: 应用于整个音频，设定整体基调
> - 内联标签: 仅作用于标签所在位置，可以局部覆盖

---

### 9. 多说话人对话

使用 `<|speaker:N|>` 标签指定不同说话人：

```python
# 双人对话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>你好，很高兴认识你。
<|speaker:1|>我也很高兴见到你。"""
    },
)

# 多角色场景
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>早上好，请问有什么可以帮您？
<|speaker:1|>我想预约明天的会议室。
<|speaker:0|>好的，请问需要多大的会议室？
<|speaker:1|>十个人左右就够了。"""
    },
)

# 带情感的对话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>[excited]你知道吗？我们公司上市了！
<|speaker:1|>[surprised]真的吗？太棒了！
<|speaker:0|>[laughing] 是的！大家一起庆祝！"""
    },
)
```

使用 curl：
```bash
# 双人对话
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=<|speaker:0|>你好，很高兴认识你。
<|speaker:1|>我也很高兴见到你。" \
  -o dialogue.wav

# 带情感的对话
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=<|speaker:0|>[excited]你知道吗？我们公司上市了！
<|speaker:1|>[surprised]真的吗？太棒了！" \
  -o dialogue_emotion.wav
```

> **说明**：
> - `<|speaker:N|>`: N 从 0 开始，相同编号的文本由同一说话人说出
> - 模型会自动将长文本按说话人和字节限制分批处理
> - 可以与情感标签组合使用

#### 多说话人音色分配（重要）

`<|speaker:N|>` 标签只解决"区分不同 speaker"的问题，**默认不显式控制每个 speaker 的音色**。N=0 和 N=1 会得到两个不同的音色，但具体是什么音色由模型决定（默认是中性的合成音色，且每次调用结果不稳定）。

**显式控制每个 speaker 音色的两种方式**：

| 方式 | 用法 | 适用场景 |
|---|---|---|
| **A. 单 speaker 模式 + per-segment 调用** | 每段对白单独调一次 API，传该角色的 `reference_audio` | 角色数固定、追求最强克隆保真度（**当前 src_next pipeline 采用**） |
| **B. 多 speaker 模式 + `reference_id` 数组**（仅云 API） | `reference_id: [id_alice, id_bob, ...]` 配合 `<\|speaker:0\|>` / `<\|speaker:1\|>` | 一次 API 调用合成多角色对话，省 RTF；要求每个 speaker 已预注册到 voice model 库 |

**自托管 wrapper 当前未暴露方式 B 的 `reference_id` 数组**——v2 计划扩展。详见第 18 节。

---

### 10. 英文合成

```python
# 基础英文
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "Hello, this is a voice synthesis test using S2-Pro."},
)

# 英文 + 情感
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[excited] This is amazing! We did it!"},
)

# 英文对话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>Good morning! How are you today?
<|speaker:1|>[happy] I'm great, thank you! And you?"""
    },
)
```

使用 curl：
```bash
# 基础英文
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=Hello, this is a voice synthesis test using S2-Pro." \
  -o english.wav

# 英文 + 情感
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[excited] This is amazing! We did it!" \
  -o english_excited.wav
```

---

### 11. 中文合成

```python
# 基础中文
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "你好，欢迎使用语音合成系统。"},
)

# 中文 + 情感
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={"text": "[happy] 今天天气真不错！"},
)

# 中文对话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>[whisper]这件事，我们要保密。
<|speaker:1|>[surprised]真的吗？
<|speaker:0|>[sigh] 是的，我也很无奈。"""
    },
)
```

使用 curl：
```bash
# 基础中文
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=你好，欢迎使用语音合成系统。" \
  -o chinese.wav

# 中文 + 情感
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=[happy] 今天天气真不错！" \
  -o chinese_happy.wav
```

---

### 12. 采样参数调优

调整采样参数控制生成质量和多样性：

```python
# 标准参数（平衡）
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "标准参数合成示例。",
        "temperature": 1.0,
        "top_p": 0.6,
    },
)

# 稳定输出（更保守）
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "稳定输出示例。",
        "temperature": 0.7,
        "top_p": 0.5,
    },
)

# 创意输出（更多变化）
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "创意输出示例。",
        "temperature": 1.2,
        "top_p": 0.9,
    },
)
```

使用 curl：
```bash
# 标准参数
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=标准参数合成示例。" \
  -F "temperature=1.0" \
  -F "top_p=0.6" \
  -o standard.wav

# 稳定输出
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=稳定输出示例。" \
  -F "temperature=0.7" \
  -F "top_p=0.5" \
  -o stable.wav

# 创意输出
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=创意输出示例。" \
  -F "temperature=1.2" \
  -F "top_p=0.9" \
  -o creative.wav
```

**参数效果对照表**：

| 场景 | temperature | top_p | 效果 |
|------|-------------|-------|------|
| 标准合成 | 1.0 | 0.6 | 平衡质量 |
| 稳定输出 | 0.7 | 0.5 | 最小变化 |
| 创意合成 | 1.2 | 0.9 | 最大变化 |

---

### 13. 音频长度控制

通过 `max_new_tokens` 控制生成音频的长度：

```python
# 短音频
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "短句测试。",
        "max_new_tokens": 1024,
    },
)

# 长音频
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "这是一段较长的语音内容，包含了更多的文字信息。",
        "max_new_tokens": 8192,
    },
)
```

使用 curl：
```bash
# 短音频
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=短句测试。" \
  -F "max_new_tokens=1024" \
  -o short.wav

# 长音频
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F "text=这是一段较长的语音内容。" \
  -F "max_new_tokens=8192" \
  -o long.wav
```

---

### 14. 组合使用示例

```python
# 综合示例：带情感的对话
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>[excited]你听说了吗？我们公司刚获得了一笔大投资！
<|speaker:1|>[surprised]真的吗？太棒了！投资了多少？
<|speaker:0|>[laughing] 一千万美元！够我们发展好几年了。
<|speaker:1|>[happy] 太好了！我们一定要好好庆祝一下！""",
        "temperature": 0.9,
        "top_p": 0.7,
    },
)

# 新闻播报风格
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": "[professional broadcast tone] 各位观众晚上好，欢迎收看今天的新闻节目。",
        "instruction": "专业播音员风格，清晰流畅",
    },
)

# 故事讲述
response = requests.post(
    "http://10.50.121.102:8006/v1/voicegen/generate",
    data={
        "text": """<|speaker:0|>[sigh] 从前，有一个小女孩住在森林边上...
<|speaker:0|>[pause] 有一天，她遇到了一只神奇的生物。""",
        "temperature": 0.8,
    },
)
```

使用 curl：
```bash
# 综合示例
curl --noproxy '*' -X POST http://10.50.121.102:8006/v1/voicegen/generate \
  -F 'text=<|speaker:0|>[excited]你听说了吗？我们公司刚获得了一笔大投资！
<|speaker:1|>[surprised]真的吗？太棒了！' \
  -F "temperature=0.9" \
  -F "top_p=0.7" \
  -o dialogue_combined.wav
```

---

## 内联标签参考表

### 情感类标签

| 标签 | 效果 | 示例 |
|------|------|------|
| `[excited]` | 兴奋、激动 | `[excited]太棒了！` |
| `[angry]` | 愤怒 | `[angry]你怎么这样！` |
| `[sad]` | 悲伤 | `[sad]一切都结束了...` |
| `[surprised]` | 惊讶 | `[surprised]真的吗？` |
| `[whisper]` | 低声耳语 | `[whisper]这是个秘密` |
| `[screaming]` | 尖叫 | `[screaming]救命！` |
| `[shouting]` | 大喊 | `[shouting]注意听！` |

### 韵律类标签

| 标签 | 效果 | 示例 |
|------|------|------|
| `[pause]` | 插入停顿 | `Hello [pause] World` |
| `[short pause]` | 短停顿 | `one [short pause] two` |
| `[emphasis]` | 强调 | `这是 [emphasis] 重点` |
| `[sigh]` | 叹气 | `[sigh] 唉...` |
| `[interrupting]` | 打断 | `[interrupting] 等一下` |

### 声音效果类标签

| 标签 | 效果 | 示例 |
|------|------|------|
| `[laughing]` | 笑声 | `哈哈 [laughing]` |
| `[chuckle]` | 轻笑 | `[chuckle] 真有趣` |
| `[inhale]` | 吸气声 | `[inhale] 呼...` |
| `[exhale]` | 呼气声 | `[exhale] 终于...` |
| `[sniff]` | 抽鼻子 | `[sniff]` |
| `[clearing throat]` | 清嗓子 | `[clearing throat]` |
| `[panting]` | 喘息 | `[panting]` |
| `[tsk]` | 啧啧声 | `[tsk] 不行` |
| `[moaning]` | 呻吟 | `[moaning]` |

### 音量类标签

| 标签 | 效果 | 示例 |
|------|------|------|
| `[loud]` | 大声 | `[loud] 注意听！` |
| `[quiet]` | 小声 | `[quiet] 听好了` |
| `[volume up]` | 提高音量 | `[volume up] 大声点` |
| `[volume down]` | 降低音量 | `[volume down] 小声点` |

### 特殊效果类标签

| 标签 | 效果 | 示例 |
|------|------|------|
| `[singing]` | 唱歌/哼唱 | `[singing] 哆来咪` |
| `[echo]` | 回声效果 | `[echo] 回声...` |
| `[with strong accent]` | 带口音 | `[with strong accent]` |
| `[delight]` | 愉悦 | `[delight]` |

---

## 注意事项

1. **--noproxy 参数**: 使用 curl 时必须加 `--noproxy '*'`
2. **Python proxies**: 使用 requests 时设置 `proxies={"http": None, "https": None}`
3. **标签语法**: 所有控制标签必须用方括号包裹，如 `[excited]`
4. **标签组合**: 可以组合使用多个标签，如 `[excited] [loud]`
5. **自由文本**: 支持自然语言描述，如 `[whisper in small voice]`
6. **超时设置**: 建议设置 300 秒以上的超时时间
7. **音频长度**: 通过 `max_new_tokens` 控制生成长度
8. **采样参数**: temperature 建议 0.7-1.2，top_p 建议 0.5-0.9
9. **多说话人**: 使用 `<|speaker:N|>` 标签，N 从 0 开始；**默认音色不可控**（详见 9.1 节）
10. **音色克隆（重要）**: 必传 `prompt_text` 且内容必须与 `reference_audio` wav **完全匹配**，否则克隆效果劣化（详见第 15 节）
11. **本地 wrapper 局限**: 自托管 `/v1/voicegen/generate` 当前只暴露 `text` / `instruction` / `temperature` / `top_p` / `max_new_tokens`；`reference_audio` / `prosody` / 输出格式等高级字段未暴露（详见第 18 节）

---

## 与其他 TTS 模型对比

| 特性 | S2-Pro | CosyVoice | IndexTTS | Qwen3-TTS |
|------|--------|-----------|----------|-----------|
| 采样率 | 44100 Hz | 24000 Hz | 22050 Hz | 24000 Hz |
| 情感控制 | 15000+ 标签 + 自由文本 | instruct 模式 | 8维向量 | 自然语言指令 |
| 内联控制 | 支持（细粒度） | 部分支持 | 不支持 | 部分支持 |
| 细粒度控制 | [pause] [emphasis] 等 | [breath] 等 | 拼音控制 | 语速/停顿 |
| 音色克隆 | **支持（reference_audio）** | 需要参考音频 | 需要参考音频 | 不需要（指令生成） |
| 语言支持 | 80+ 种语言 | 9种语言 + 方言 | 中英文 | 10种语言 |
| 多说话人 | 支持 | 支持 | 支持 | 支持 |
| 说话方式 | [singing] [whisper] 等 | limited | 不支持 | limited |

---

## 15. 音色克隆（Voice Cloning）

S2Pro **完全支持音色克隆**（zero-shot voice cloning）。机制是把一段 10-30 秒的参考音频（`reference_audio`）连同其转录文本（`prompt_text`）一起送给模型，模型会从参考音频中提取音色特征（timbre / speaking style / emotion），应用到目标文本的合成上。

### 15.1 必备参数（三件套）

| 参数 | 说明 | 必填 |
|---|---|---|
| `reference_audio` | 参考音频文件（wav / mp3），10-30 秒，干净无噪声 | ✅ |
| `prompt_text` | 参考音频的**精确转录文本**（必须与 wav 内容一致） | ✅ |
| `enable_reference_audio` | 开启克隆模式的标志位（设为 `true`） | ✅ |

> ⚠️ **关键约束**：`prompt_text` 必须与 `reference_audio` 的实际语音内容**逐字匹配**，否则模型会把"不匹配的文本 + 音频"当噪声处理，克隆效果严重劣化。参考 fish-speech Issue [#836](https://github.com/fishaudio/fish-speech/issues/836)。

### 15.2 参考音频要求

| 维度 | 推荐 |
|---|---|
| 时长 | **10-30 秒**（过短音色捕捉不全，过长编码耗时） |
| 质量 | 干净单语音，无背景噪声 / 音乐 / 回声 |
| 采样率 | ≥ 16 kHz（44.1 kHz / 48 kHz 最佳） |
| 格式 | WAV（16-bit PCM）或 MP3 |
| 内容 | 单一说话人，语速稳定，避免长沉默 |

### 15.3 单说话人克隆示例（云 API 格式）

```python
# 云 API：POST https://api.fish.audio/v1/tts
# 用 reference_id（已预注册的 voice model id）
response = requests.post(
    "https://api.fish.audio/v1/tts",
    headers={
        "Authorization": "Bearer <token>",
        "Content-Type": "application/json",
        "model": "s2-pro",
    },
    json={
        "text": "今天天气真不错，我们一起去公园散步吧。",
        "reference_id": "model-id-alice",  # 预注册的 voice model id
        "temperature": 0.7,
        "top_p": 0.7,
    },
    timeout=300,
)
```

或用 `references` 数组做 inline 零样本克隆（无需预注册）：

```python
import base64

with open("alice_reference.wav", "rb") as f:
    audio_bytes = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://api.fish.audio/v1/tts",
    headers={
        "Authorization": "Bearer <token>",
        "Content-Type": "application/json",
        "model": "s2-pro",
    },
    json={
        "text": "今天天气真不错。",
        "references": [{
            "audio": audio_bytes,                       # base64 编码的 wav
            "text": "这是 alice 的参考音频转录文本。",   # 必须与 wav 内容匹配
        }],
        "temperature": 0.7,
    },
    timeout=300,
)
```

### 15.4 多说话人 + 各自克隆（云 API）

S2-Pro 独有：多 speaker 模式下，可以给每个 speaker 显式指定参考音频：

```python
response = requests.post(
    "https://api.fish.audio/v1/tts",
    headers={
        "Authorization": "Bearer <token>",
        "Content-Type": "application/json",
        "model": "s2-pro",
    },
    json={
        "text": (
            "<|speaker:0|>你好，很高兴认识你。"
            "<|speaker:1|>我也很高兴见到你。"
        ),
        "reference_id": ["model-id-alice", "model-id-bob"],  # 数组：每个 speaker 一个 id
        # 或用 references 2D 数组做 inline：
        # "references": [[alice_ref], [bob_ref]],
        "temperature": 0.7,
    },
    timeout=300,
)
```

### 15.5 自托管 wrapper 的音色克隆支持

**当前 `/v1/voicegen/generate` 未暴露 `reference_audio` 字段**。若需在自托管服务器上做音色克隆，有两种路径：

| 路径 | 改动 | 适用场景 |
|---|---|---|
| **A. 扩展 wrapper** | 修改 `servers/api_server_s2pro.py` 接受 multipart `reference_audio` 文件 + `prompt_text` 字段，转调底层 fish-speech 的 `encode_reference()` + `enable_reference_audio=true` | 生产用；推荐 |
| **B. 直接调上游 `/v1/tts`** | 绕过 voicegen wrapper，直接调 fish-speech 官方 api_server 暴露的 `/v1/tts` 端点（若服务器开了） | 临时调试用 |

底层 fish-speech 源码已实现 reference audio 支持（见 [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)），wrapper 只是没有把这套参数透传出来。

### 15.6 跨调用一致性 & 性能

- **每次调用无状态**：reference_audio 每次都被重新编码（VQ tokenize），不存在 session 级缓存。这意味着 100 段相同角色的合成会做 100 次编码——是 RTF 瓶颈。参考 Discussion [#1300](https://github.com/fishaudio/fish-speech/discussions/1300)。
- **同一段落内多次调用音色一致**：因为相同 reference_audio + 相同 prompt_text 会产生稳定音色特征。
- **跨调用音色基本一致但非完全相同**：合成路径含 temperature/top_p 采样，会有微小波动。需要完全一致时把 temperature 设为 0。

---

## 16. 输出格式与采样率

### 16.1 支持的输出格式

| 格式 | 采样率 | 默认采样率 | 比特率 / 位深 | 备注 |
|---|---|---|---|---|
| WAV / PCM | 8 / 16 / 24 / 32 / 44.1 kHz | **44.1 kHz** | 16-bit mono | 无损；本地 wrapper 默认 |
| MP3 | 32 / 44.1 kHz | **44.1 kHz** | 64 / 128（默认） / 192 kbps | 体积小，适合分发 |
| Opus | 48 kHz | **48 kHz** | auto / 24 / 32（默认） / 48 / 64 kbps | 低延迟直播首选 |

### 16.2 设置方式（云 API）

```python
response = requests.post(
    "https://api.fish.audio/v1/tts",
    headers={"Authorization": "Bearer <token>", "model": "s2-pro"},
    json={
        "text": "...",
        "format": "mp3",          # wav / pcm / mp3 / opus
        "sample_rate": 44100,     # None = 用 format 默认
        "mp3_bitrate": 128,       # 仅 format=mp3 时生效：64/128/192
        "opus_bitrate": 32000,    # 仅 format=opus 时生效：-1000(auto)/24k/32k/48k/64k
    },
)
```

### 16.3 自托管 wrapper 局限

`/v1/voicegen/generate` 当前**只输出 44100 Hz WAV**。要支持 mp3/opus/pcm 需扩展 wrapper 接受 `format` / `sample_rate` 字段，并修改响应头 Content-Type。

---

## 17. 韵律与高级生成参数

S2Pro 除了内联标签（`[pause]` / `[emphasis]` 等）做局部韵律控制外，云 API 还提供**全局参数级**的韵律与生成控制：

### 17.1 参数总览

| 参数 | 类型 | 默认 | 范围 | 作用 |
|---|---|---|---|---|
| `prosody.speed` | float | 1.0 | 0.5-2.0 | 全局语速倍率（乘性）；与 `[speak slowly]` / `[speak quickly]` 内联标签效果叠加 |
| `prosody.volume` | float | 0 | -10-+10 dB | 全局音量调整；正数变大，负数变小 |
| `prosody.normalize_loudness` | bool | true | - | 响度归一化，避免不同段音量跳变 |
| `chunk_length` | int | 300 | 100-300 | 文本分块字符数；长文本会自动切块合成再拼接 |
| `min_chunk_length` | int | 50 | 0-100 | 最小分块字符数；防止过短导致语流不连贯 |
| `condition_on_previous_chunks` | bool | true | - | 跨块时保留上一块结尾作上下文，提升音色 / 韵律一致性 |
| `repetition_penalty` | float | 1.2 | 1.0-2.0 | 重复音频模式惩罚；>1.0 减少重复（卡音、复读） |
| `early_stop_threshold` | float | 1.0 | 0-1 | 批量合成的早停阈值；越低越激进 |
| `latency` | enum | normal | normal / balanced / low | 延迟-质量权衡；`low` 适合流式直播 |
| `normalize` | bool | true | - | 文本归一化（数字 / 英文 / 中文标点稳定化） |

### 17.2 推荐组合

| 场景 | prosody.speed | prosody.volume | chunk_length | latency | 备注 |
|---|---|---|---|---|---|
| 有声书旁白 | 0.95 | 0 | 300 | normal | 略慢，长块保连贯 |
| 角色对白 | 1.0 | 0 | 200 | normal | 让内联标签发挥 |
| 直播字幕 | 1.0 | 0 | 100 | low | 最低延迟 |
| 新闻播报 | 1.05 | +2 | 300 | normal | 偏快偏响 |

### 17.3 自托管 wrapper 局限

`/v1/voicegen/generate` 当前**只暴露 `temperature` / `top_p` / `max_new_tokens`**。`prosody` / `chunk_length` / `latency` 等需扩展 wrapper。

---

## 18. 云 API vs 自托管 wrapper 对比

| 维度 | 云 API `api.fish.audio/v1/tts` | 自托管 wrapper `10.50.121.102:8006/v1/voicegen/generate` |
|---|---|---|
| 鉴权 | Bearer token | 无（内网） |
| 协议 | JSON body | multipart/form-data |
| **音色克隆** | `reference_id` 字符串 / `references` 数组 | **未暴露**（v2 待扩展 wrapper） |
| 多说话人 + per-speaker 音色 | `reference_id: [id1, id2]` 数组 + `<\|speaker:N\|>` | 仅 `<\|speaker:N\|>`（音色随机） |
| 内联标签 | `[tag]` 全支持 | 同 |
| 全局 instruction | 支持 | 支持 |
| 输出格式 | wav / pcm / mp3 / opus | 仅 wav |
| 采样率 | 8/16/24/32/44.1/48 kHz | 固定 44.1 kHz |
| Prosody 控制 | `prosody: {speed, volume, normalize_loudness}` | 未暴露 |
| Latency 模式 | normal / balanced / low | 未暴露 |
| Chunk 控制 | `chunk_length` / `min_chunk_length` / `condition_on_previous_chunks` | 未暴露 |
| Repetition penalty | 支持 | 未暴露 |
| 计费 | 按 character 计费 | 免费（自托管） |
| 模型版本 | 始终最新（s2-pro） | 部署版本（s2-pro 4B） |

**何时用哪个**：
- 开发 / 调试 / 离线生产 → 自托管（免费、可控）
- 需要云独有功能（多 speaker + per-speaker id、低延迟直播、mp3 直出） → 云 API
- 混合用：profile 切换即可（src_next 已支持 `s2pro_http` backend，仅需改 `base_url`）

---

## 19. 自托管 wrapper 已知限制 & v2 路线图

### 19.1 当前 wrapper 缺失字段

按优先级排序（影响从大到小）：

1. **`reference_audio` + `prompt_text` + `enable_reference_audio`**：阻塞音色克隆功能（src_next pipeline 的核心需求）
2. **`reference_id` 数组**：阻塞多角色显式音色分配
3. **`format` / `sample_rate`**：阻塞 mp3 直出（节省磁盘）
4. **`prosody` 字段**：阻塞全局韵律控制（目前只能靠内联标签）
5. **`latency` 模式**：阻塞低延迟流式合成
6. **`chunk_length` / `condition_on_previous_chunks`**：阻塞长文本优化

### 19.2 v2 扩展路线图

| 版本 | 目标 | 改动范围 |
|---|---|---|
| v1（当前） | 基础合成 + 内联标签 + 全局 instruction | 已就绪 |
| v1.5 | 透传 `reference_audio` 三件套 | 改 `api_server_s2pro.py` |
| v2.0 | 透传 `prosody` / `chunk_length` / `format` | 改 wrapper + 加 profile 字段 |
| v2.5 | 多 speaker + `reference_id` 数组 | wrapper 改造 + adapter 拼接逻辑 |
| v3.0 | reference_audio 编码缓存 | wrapper 加 session 字段 |

### 19.3 参考文档

- 官方 GitHub：[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- 云 API Reference：[docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech)
- 自托管推理指南：[docs.fish.audio/developer-guide/self-hosting/running-inference](https://docs.fish.audio/developer-guide/self-hosting/running-inference)
- 语音克隆最佳实践：[docs.fish.audio/developer-guide/best-practices/voice-cloning](https://docs.fish.audio/developer-guide/best-practices/voice-cloning)
- 技术报告：[arxiv.org/html/2603.08823](https://arxiv.org/html/2603.08823v1)
- Issue #836（克隆保真度）：[github.com/fishaudio/fish-speech/issues/836](https://github.com/fishaudio/fish-speech/issues/836)
- Discussion #639（随机 speaker 问题）：[github.com/fishaudio/fish-speech/discussions/639](https://github.com/fishaudio/fish-speech/discussions/639)
- Discussion #1300（无状态 API 延迟）：[github.com/fishaudio/fish-speech/discussions/1300](https://github.com/fishaudio/fish-speech/discussions/1300)