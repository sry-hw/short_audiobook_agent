# S2Pro TTS Adapter 设计理念

> File: `src_next/tts/s2pro_adapter.py`
> Backend: `s2pro_http`
> Model: Fish Audio S2-Pro 4B
> Stage: **v1 转换验证**（不调真实 HTTP）

---

## 1. 定位：控制信号增强层 + 音色路由层

本 adapter **不是**简单 TTS wrapper。它在通用 `TTSInstruction` 和 S2Pro API 之间做**两层翻译**：

### Layer 1：风格控制翻译（通用字段 → S2Pro 内联标签）

S2Pro 支持 **15000+ 内联标签**（`[excited]` / `[pause]` / `[emphasis]` / `[laughing]` / `[pitch up]` 等）+ 自由文本描述（`[whisper in small voice]` / `[professional broadcast tone]`）。这些标签是 S2Pro 区别于 CosyVoice / IndexTTS 的核心能力。

本 adapter 把通用字段翻译成标签：

```
通用 TTSInstruction                    S2Pro 内联标签
─────────────────────                ─────────────────────
emotion=sad                  →       [sad]
emotion_intensity=0.7        →       [sigh]  （high density 时）
pace=0.85                    →       [speak slowly]
volume=soft                  →       [quiet]
pitch=medium_low             →       [pitch down]
pause_hint=0.9               →       [pause]  （段末追加）
stress_words=["松果"]         →       在原文中包裹 [emphasis]松果
delivery_instruction="..."   →       全局 instruction 字段（不内联）
```

### Layer 2：音色路由翻译（voicebank → reference_audio）

S2Pro 支持 **音色克隆**（zero-shot voice cloning）：传 `reference_audio`（10-30 秒 wav）+ `prompt_text`（精确转录）+ `enable_reference_audio=true`，模型会克隆参考音色。

本 adapter 从 `voicebank_result.speaker_to_voice[speaker]` 查 wav 路径，写入 `S2ProRenderResult.reference_audio_path`。voicebank 层完全复用 `qwen3_http` VoiceDesign。

---

## 2. 与 CosyVoice / IndexTTS 的差异

| 维度 | S2Pro adapter | CosyVoice adapter | IndexTTS adapter |
|---|---|---|---|
| 控制信号载体 | `[tag]` 内联 + instruction 全局 | 自然语言 prompt_text + `<\|endofprompt\|>` | 8 维 emotion_vector |
| 风格翻译层数 | 7 步（emotion/pace/volume/pitch/pause/stress/inferred） | 1 步（合并成 prompt_text） | 1 步（合并成 vector） |
| 音色克隆字段 | `reference_audio` + `prompt_text` | `prompt_audio`（base64） | `prompt_audio`（路径） |
| 标签密度可控 | ✅ low/medium/high | ❌ 全打包 | ❌ 全打包 |
| 自由文本描述 | ✅（`[whisper in small voice]` 等） | ✅（prompt_text 自由文本） | ❌ |

**结论**：S2Pro 的表达力最强，但也最复杂。adapter 必须做密度控制（`max_tag_density`），避免标签过密导致模型困惑。

---

## 3. 转换规则总览

详细规则见 plan 文件 `melodic-munching-engelbart.md` 第 4 节。关键表：

### emotion → tag

| DirectorInstruction.emotion | S2Pro 标签 |
|---|---|
| excited / sad / angry / surprised / fearful | S2Pro 固定标签（`[excited]` 等） |
| happy / joyful / calm / nostalgic / moved | 自由文本（`[happy]` / `[calm]` 等） |
| warm / gentle / playful / serious / anxious | 自由文本描述（`[warm tone]` / `[gentle voice]` 等） |
| neutral / unknown | 留空（omit） |

### pace / volume / pitch → tag

```
pace  ≤ 0.90  →  [speak slowly]
pace  ≥ 1.10  →  [speak quickly]

volume=soft   →  [quiet]
volume=strong →  [loud]

pitch=low/medium_low        →  [pitch down]
pitch=medium_high/high      →  [pitch up]
```

### pause_hint → tag（段末追加）

```
pause_hint ≥ 0.8    →  [pause]
pause_hint 0.4-0.8  →  [short pause]
pause_hint < 0.4    →  omit
```

### stress_words → inline `[emphasis]`

原文 `小松鼠背着一袋松果从树上跳下来` + stress_words=`["松果"]`
→ `小松鼠背着一袋[emphasis]松果从树上跳下来`

### emotion_intensity → 推断标签（仅 `max_tag_density=high`）

| 触发 | 标签 | 位置 |
|---|---|---|
| sad + intensity ≥ 0.7 | `[sigh]` | prepend |
| anxious + intensity ≥ 0.7 | `[inhale]` | prepend |
| moved + intensity ≥ 0.7 | `[exhale]` | prepend |
| angry + intensity ≥ 0.85 | `[shouting]` | prepend |
| joyful / excited + intensity ≥ 0.85 | `[laughing]` | append |

---

## 4. 标签密度三档对比

`profile.yaml` 中 `tts.extra_args.max_tag_density` 控制：

| density | 启用的标签 | 用例 |
|---|---|---|
| `low` | emotion + pause + stress_wraps（最干净） | 测试 / 出错排障；最小干预 |
| `medium`（默认） | + pace + volume + pitch | 生产推荐；平衡表达力与稳定性 |
| `high` | + intensity-driven inferred tags | 角色戏精场景；强调情感爆发 |

**输出对比**（emotion=sad, intensity=0.7, pace=0.85, volume=soft, pitch=medium_low, pause_hint=1.0, stress_words=`["故乡"]`）：

```
low:     "[sad] 望着远方的[emphasis]故乡，我心中泛起阵阵思念。[pause]"
medium:  "[sad][speak slowly][quiet][pitch down] 望着远方的[emphasis]故乡，...[pause]"
high:    "[sigh][sad][speak slowly][quiet][pitch down] 望着远方的[emphasis]故乡，...[laughing][pause]"
```

---

## 5. 音色克隆路由

### 数据流

```
voicebank 层（qwen3_http VoiceDesign）
    生成 <output>/voicebank/<speaker>.wav
                ↓
voicebank_result.speaker_to_voice = {"小松鼠": ".../voicebank/小松鼠.wav", ...}
                ↓
S2ProTTSAdapter._resolve_reference_audio(instruction, voicebank_result)
    1. 优先：speaker_to_voice[instruction.speaker]
    2. 兜底 1：instruction.voice_ref（tts_instruction_builder 已填）
    3. 兜底 2：speaker_to_voice["narrator"]
    4. 兜底 3：空字符串（reference_audio 不可用）
                ↓
S2ProRenderResult.reference_audio_path = 查到的 wav 路径
S2ProRenderResult.enable_reference_audio = True / False
S2ProRenderResult.prompt_text = profile.tts.extra_args.reference_text
```

### 关键约束

1. **`prompt_text` 必须与 voicebank wav 内容匹配**：
   - voicebank 用 "你好，欢迎使用语音设计功能，这是一个测试句子。" 合成 wav
   - S2Pro 也必须传同样的 prompt_text
   - 否则克隆效果劣化（参考 fish-speech Issue [#836](https://github.com/fishaudio/fish-speech/issues/836)）
   - profile 中 `tts.reference_text` 应与 `voicebank.reference_text` 完全一致

2. **API 无状态**：每次调用都重新编码 reference_audio；100 段相同角色 = 100 次编码。v3 考虑加 caching。

3. **本地 wrapper 局限**：自托管 `/v1/voicegen/generate` 当前**未暴露 reference_audio 字段**。v2 真实合成前需扩展 wrapper（详见 `usage_guide_s2pro.md` 第 18-19 节）。

---

## 6. S2ProRenderResult 结构

```python
@dataclass
class S2ProRenderResult:
    segment_id: str               # 段落 ID
    speaker: str                  # 说话人
    original_text: str            # 原文（未动）
    s2pro_text: str               # 含内联标签的文本（→ API text 字段）
    instruction: str              # 全局风格（→ API instruction 字段）
    reference_audio_path: str     # voicebank wav 路径（→ API reference_audio）
    enable_reference_audio: bool  # 启用克隆标志
    prompt_text: str              # 参考音频转录（→ API prompt_text）
    params: dict                  # temperature / top_p / max_new_tokens
    debug_tags: dict              # 转换诊断（哪些标签被加、来源、跳过原因）
```

---

## 7. v1 局限与未来扩展

### v1 已实现

- ✅ 7 步转换逻辑（emotion / pace / volume / pitch / pause / stress / inferred）
- ✅ 三档密度控制
- ✅ reference_audio 路由（4 级 fallback）
- ✅ 转换结果落盘（`<audio_dir>/<seg>.s2pro.txt` + `s2pro_render_results.json`）
- ✅ debug_tags 完整诊断

### v1 未实现（v2 路线）

- ❌ 真实 HTTP 合成（需先扩展本地 wrapper 接受 reference_audio）
- ❌ 句子内部 pause 位置（当前只能段末追加，无法精确到字符级）
- ❌ 多说话人拼接（`<|speaker:N|>` + 单次 API 调用多角色，省 RTF）
- ❌ reference_audio 编码缓存

### 扩展点：metadata["s2pro_hints"]

未来若需更精细控制（如句子内部 pause 位置、自定义 breath 插入点），可通过 `TTSInstruction.metadata["s2pro_hints"]` 注入：

```python
# 未来 v2 支持（当前 adapter 还未读取此字段）
TTSInstruction(
    ...,
    metadata={
        "s2pro_hints": {
            "inline_pauses": [12, 35],   # 在原文 char index 12 / 35 处插入 [pause]
            "sigh_before": True,          # 段首加 [sigh]
            "custom_tags": ["[whisper]"], # 用户自定义追加
        }
    }
)
```

无需修改 DirectorInstruction / TTSInstruction schema；高级用户通过 metadata 注入。

---

## 8. 验证

### 单元级（不调 LLM）

```bash
python -c "
from src_next.tts.s2pro_adapter import S2ProTTSAdapter
from src_next.core.data_models import TTSInstruction
inst = TTSInstruction(
    segment_id='seg_test', speaker='narrator', segment_type='narration',
    text='望着远方的故乡，我心中泛起阵阵思念。',
    emotion='sad', emotion_intensity=0.7,
    pace=0.85, tone='serious', volume='soft', pitch='medium_low',
    pause_hint=1.0, stress_words=['故乡'],
    delivery_instruction='以低沉缓慢的语气叙述，传递感伤和沉重。',
)
a = S2ProTTSAdapter(base_url='http://x', extra_args={'max_tag_density': 'high'})
r = a.convert_instruction(inst)
print(r.s2pro_text)
"
```

预期输出含 `[sigh][sad][speak slowly][quiet][pitch down]` 前缀 + `[emphasis]故乡` + 末尾 `[pause]`。

### 端到端（蓝区 qwen3 LLM）

```bash
python src_next/tts/test_s2pro_adapter.py
```

预期：
- 6 步 analysis 完成（~30s，取决于 LLM 速度）
- 控制台打印每条 segment 的转换对比
- 落盘 `output-src-next/s2pro-adapter-test/sample_story_01/s2pro_render_results.json`
- 不同 segment 的 emotion / pace / volume / pitch 差异在 s2pro_text 中可观察

### 切换密度

修改 profile 的 `max_tag_density: low`，重跑 test，观察：
- `pace_tag` / `volume_tag` / `pitch_tag` 从 s2pro_text 消失
- `inferred_pre_tags` / `inferred_post_tags` 为空

---

## 9. 参考

- 项目内：`usage_guide_s2pro.md`（本地 wrapper 文档 + 第 15-19 节云 API / 克隆 / 对比 / 限制）
- 官方 GitHub：[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- 官方 API Reference：[docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech)
- 自托管指南：[docs.fish.audio/developer-guide/self-hosting/running-inference](https://docs.fish.audio/developer-guide/self-hosting/running-inference)
- 克隆保真度 Issue：[#836](https://github.com/fishaudio/fish-speech/issues/836)
- 随机 speaker 问题 Discussion：[#639](https://github.com/fishaudio/fish-speech/discussions/639)
- 无状态 API 延迟 Discussion：[#1300](https://github.com/fishaudio/fish-speech/discussions/1300)
