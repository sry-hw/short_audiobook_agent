# analysis/ 语义分析层

> 本层是 `src_next/`` 重构架构中的语义理解层。整体架构图、各层关系、数据流请见 [../README.md](../README.md) 的「〇、一图看懂 src_next」。

## 一、这一层负责什么

把 core 层切出来的纯文本 segments，经过 LLM 分析，产出后续 voicebank / tts / audio 层需要的语义结构：

- **引号内容分类**：判断每个引号内容是真对白、心理活动，还是强调词 / 书名 / 术语（非对白引号并回 narration）。
- **说话人识别**：对真正的对白 / 心理活动 segment，判断是谁说的。
- **角色档案生成**：从故事中提取角色清单，为每个角色生成 gender / age_style / personality / voice_prompt。
- **导演计划生成**：为每个 segment 生成 emotion / pace / tone / pause_hint / delivery_instruction。
- **结构稳定性**：把 LLM 的随机输出清洗成符合 `src_next.core.data_models` 的 dataclass 实例。

## 二、这一层不负责什么

- **不直接发 HTTP**：所有 LLM 调用通过 `BaseLLMClient.generate_json`。
- **不绑定具体后端**：不 import `QwenHTTPClient` / `Gemma4HTTPClient`，切换后端零改动。
- **不生成音频**：TTS / voicebank / audio_merger 都不在本层。
- **不读 .env / 不读 profile yaml**：环境配置由上层注入。
- **不写文件**：中间产物持久化由 core pipeline 负责。
- **不做整体风格分析（v1 简化）**：旧 src 的 overall_style / genre 推断暂未搬过来。

## 三、四个文件的输入输出

### 3.1 `quote_classifier.py`

```python
def classify_and_merge_quotes(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
    output_debug_path: str | None = None,
) -> list[Segment]:
```

- **输入**：`core.segment_builder.build_segments()` 输出的 Segment 列表（所有引号都已经切成 `segment_type="dialogue", speaker="unknown"` 的候选段）。
- **输出**：新的 Segment 列表（深拷贝，**长度可能 ≤ 输入**）。非对白引号已并回相邻 narration，dialogue/inner_thought 保留为独立 segment。所有 segment_id 重新编号。
- **契约**：允许 N → M（M ≤ N）。

策略：

1. **按段落分组调 LLM**：把所有 `segment_type="dialogue" AND speaker="unknown"` 的候选段按 `raw_index` 分组，每组用拼回的段落原文 + 候选列表丢给 LLM。
2. **只用 LLM 判断**：quote_type 完全交给 LLM，不做规则识别。LLM 返回 `dialogue / inner_thought / quoted_term / title_or_name / unknown` 五类之一。
3. **合并规则**：
   - `dialogue / inner_thought` → 保留独立 segment（segment_type 改成 quote_type）。
   - `quoted_term / title_or_name / unknown` → 在同段落内拼到相邻 narration，加回中文引号字符；同段落若有多段 narration 被夹在合并型 candidate 之间，会自动接成一段。
4. **失败 fallback**：LLM 调用失败 / 返回结构异常 → 本段所有 candidate 保守按 `unknown` 处理，全部并回 narration；debug JSON 里 `source="fallback"`。

### 3.2 `story_resolver.py`

```python
def resolve_speakers(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[Segment]:
```

- **输入**：通常是 `quote_classifier.classify_and_merge_quotes()` 的输出。`segment_type="dialogue"` 或 `"inner_thought"` 的 segment 的 speaker 应为 `"unknown"`，等待本函数填充。
- **输出**：新的 Segment 列表（深拷贝，入参不变；**长度严格等于输入，1:1**）。每个 dialogue / inner_thought segment 的 speaker 被填成 LLM 判定的角色名；narration segment 不动。
- **契约**：**1:1，不增删 Segment**。LLM 判不出 speaker 时填 `narrator`，但 segment 保留（不 merge）。归并痕迹不写回 Segment，只存最终 speaker；真正的别名合并交给 `character_analyzer`。

策略：

1. **按段落分组**：把 segments 按 `raw_index` 分组。
2. **每组单独调 LLM**：
   - 跳过没 dialogue / inner_thought 的组（纯旁白段落）。
   - 否则用组内全部 segment 按原顺序拼回近似段落原文（dialogue / inner_thought 段加回引号字符）作为上下文，连同候选列表丢给 LLM。
3. **写回 speaker**：把 LLM 返回的 speaker 写到对应 segment 上，过一次 `_normalize_speaker_alias` 做最保守归并（清洗标点 + 全等命中，不做后缀剥离）。
4. **跨段角色一致性**：维护 `known_speakers: list[str]` 跨段累积。每段调 LLM 时把已识别角色一并传入 prompt，强约束 LLM 优先复用已知名，减少同一角色在不同段落里命名漂移。
5. **fallback**：LLM 失败 / 没覆盖到的 → `speaker=narrator`，但 segment 保留。

> **本层只处理 dialogue / inner_thought，不判断引号是不是对白**。后者是 `quote_classifier.py` 的职责。如果跳过 quote_classifier 直接调 resolve_speakers，所有 dialogue candidate 都会被当成真对白处理（包括书名、强调词等），产出会偏。

### 3.3 `character_analyzer.py`

```python
def analyze_characters(
    segments: list[Segment],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[CharacterProfile]:
```

- **输入**：resolved segments（speaker 已识别）+ 故事全文（推荐传全文而非仅标题）。
- **输出**：`CharacterProfile` 列表。`index=0` 永远是 narrator，其余按 canonical name 在 segments 中首次出现位置排序。同一角色的多个称呼已合并到一个 CharacterProfile，被合并的称呼写进 `aliases` 字段。

字段约束：

- **narrator** 走 LLM 动态分析（基于全文判断文体风格 → 推荐旁白音色）；失败 / 全文太短 → fallback 到默认档案（`female / young / "温柔亲切的年轻女性声音，语气平稳，富有讲故事的感觉"`）。`aliases=[]`。
- **普通角色** 由 LLM 生成；`voice_prompt` 必须是 **Qwen3-TTS VoiceDesign 期望的自然语言描述性短语**（**不是「用...说」格式**——usage_guide_qwen3.md 没有此要求，且「嗓音」「说」这种词会让 Qwen3-TTS 模型困惑导致性别错乱）。长度 10~50 字，必须包含 4 个维度：
  - 性别（男 / 女 / 中性）
  - 年龄感（童 / 青年 / 中年 / 老年）
  - 音色特征（清亮 / 沙哑 / 磁性 / 柔和 / 苍老 / 清脆 ...）
  - 情绪或语气（亲切 / 严肃 / 活泼 / 温暖 / 平稳 ...）

  参考写法（`character_analyzer.py:_CHARACTER_SYSTEM_PROMPT` 里也有）：
  - `温柔关切的中年女性声音，语速平稳，带着母爱的温暖`
  - `低沉磁性的中年男性声音，语速缓慢，成熟稳重`
  - `苍老沙哑的老年声音，语速偏慢，富有沧桑感`
- **aliases**：由 LLM 在归并阶段填入；不含 canonical 本身；narrator 永远空。
- **LLM 失败时** 根据角色名关键词（动物 / 老人 / 儿童）走 fallback，confidence 标低（0.3~0.4）；fallback 也输出 Qwen3 描述性短语格式。

角色归并策略：

1. **LLM 输出 alias_map**：prompt 要求 LLM 输出 `characters` 数组 + `alias_map` 字典（别名 → canonical 名）。canonical 名优先采用原文首次出现且稳定的称呼（优先级：人名 > 职务 > 代词）。
2. **LLM 缺失 alias_map 时走 fallback**：用本文件内的 `_SAFE_TITLES`（先生/女士/老师/师傅/同志/小姐/大叔/阿姨/爷爷/奶奶 10 个安全称谓）做后缀剥离。
3. **合并**：把别名 speaker 折叠到 canonical，累积到 `CharacterProfile.aliases`。
4. **不合并的边界**：父子关系（"小明" vs "小明爸爸"）、主仆关系、人和动物不得当作同一角色。
5. **排序**：按 canonical name 在 segments 中首次出现位置 `(raw_index, segment_id)` 排序。

详见第九节「角色归并设计」。

### 3.4 `story_director.py`

```python
def generate_director_plan(
    segments: list[Segment],
    characters: list[CharacterProfile],
    llm_client: BaseLLMClient,
    *,
    story_context: str = "",
) -> list[DirectorInstruction]:
```

- **输入**：resolved segments + characters 列表。
- **输出**：`DirectorInstruction` 列表，长度严格等于 segments，按 segment 顺序排列。
- **定位**：**通用语义导演层**，不绑定任何具体 TTS 后端。后续
  `core/tts_instruction_builder.py` 会把这些字段翻译成各 TTS adapter
  能消费的具体参数或 prompt。

字段约束（11 个）：

- `emotion`：`neutral / warm / happy / excited / nostalgic / sad / gentle / anxious / playful / serious / moved / surprised / calm / joyful / longing` 等。
- `emotion_intensity`：0.0~1.0 浮点数。
- `pace`：0.75~1.30 浮点数，1.0 为正常速。
- `tone`：`gentle / warm / serious / playful / calm / lively / normal` 等。
- `volume`：`soft / normal / strong`。
- `pitch`：`low / medium_low / medium / medium_high / high`。
- `pause_hint`：0.4~1.5 秒（v2 提高范围，避免中文 TTS 输出连贯 + 短 pause 导致段间紧凑）；段内对白切换 ≥ 0.5 秒；段落结尾 ≥ 0.8 秒；童话 / 儿童故事可适度延长。`audio_merger` 还有 0.4 秒最小静音下限双保险。
- `stress_words`：1~3 个原文关键词。
- `delivery_instruction`：10~50 字中文，必须结合原文内容；**禁止** "自然对白语气" / "平稳叙述" 等空泛表达。

Fallback 增强：
- **narration**：按文本关键词细分（含 "故乡/童年/想起" → nostalgic + soft；含 "快乐/喜欢" → joyful + lively；默认 calm）。
- **dialogue**：按角色年龄段分流。儿童 → playful/excited + medium_high pitch；长辈 → gentle/warm + medium_low pitch；其他按台词标点推断。
- LLM 给的 `delivery_instruction` 若命中空泛短语黑名单（如 "自然对白语气"），会被 fallback 重写。

## 四、为什么只能依赖 BaseLLMClient

- analysis 层是「业务逻辑」层，应该和具体的 LLM 后端（Qwen / Gemma4 / 本地模型）解耦。
- 切换后端只需要在 app 层换一个 `BaseLLMClient` 实例，analysis 层代码零改动。
- `MockLLMClient` 可以在不访问网络的情况下验证 analysis 数据流（CI / 本地无 GPU 环境），直接 `llm = MockLLMClient()` 即可，不需要 `.env` 配置。
- 四个函数都做了 MockLLM 返回结构（默认占位 dict）的兼容：检测到不匹配的形状时走 fallback，不会抛异常。

## 五、和其他层的交互

```text
                ┌──────────────────────────────────────┐
                │   core/segment_builder.py            │
                │   (paragraph split + quote split)    │
                └─────────────┬────────────────────────┘
                              │ Segment[]（所有引号都先切成 dialogue 候选，
                              │           speaker=unknown）
                              ▼
              ┌─────────────────────────────────────────┐
              │  analysis/quote_classifier.py           │
              │  (按 raw_index 分组 → 每段一次 LLM)      │
              │  判 quote_type：dialogue / inner_thought│
              │  / quoted_term / title_or_name / unknown│
              └─────────────┬───────────────────────────┘
                            │ Segment[]（N→M，非对白引号已并回 narration）
                            ▼
              ┌─────────────────────────────────────────┐
              │  analysis/story_resolver.py             │
              │  (按 raw_index 分组 → 每段一次 LLM,      │
              │   跨段累积 known_speakers)               │
              │  只对 dialogue / inner_thought 问 speaker│
              └─────────────┬───────────────────────────┘
                            │ Segment[] (1:1，speaker 已填)
              ┌─────────────┴───────────────────────────┐
              ▼                                         ▼
  ┌────────────────────────────┐    ┌─────────────────────────────┐
  │ analysis/                  │    │  analysis/                  │
  │ character_analyzer.py      │    │  story_director.py          │
  │ (含 alias_map 归并)        │    │                              │
  └─────────────┬──────────────┘    └─────────────┬───────────────┘
                │ CharacterProfile[]                 │ DirectorInstruction[]
                ▼                                    ▼
       voicebank/                           core/tts_instruction_builder.py
       (prepare_voicebank)                  (build_tts_instructions)
```

- **上游**：`core/segment_builder.py`（提供已经做过段落+引号两级切分的 `Segment[]`）。
- **下游**：
  - `CharacterProfile[]` → `voicebank/`（生成参考音频）。
  - `DirectorInstruction[]` → `core/tts_instruction_builder.py`（合成 TTS 指令）。
- **同级依赖**：`llm/`（只通过 `BaseLLMClient` 接口）。

## 六、参考旧 `src` 的地方

| 旧 src 文件 | 借鉴点 | 改动点 |
|---|---|---|
| `src/story_parser.py` | 引号切分的 regex 思路（`_QUOTE_PAIRS` + `_extract_parts`） | 已合并到 `src_next/core/segment_builder.py`，不再单独建文件 |
| `src/llm_story_resolver.py` | speaker 识别的 prompt 设计；按段落批量调 LLM 的思路 | 不再做"段落 → part"二级 dict 结构；直接按 raw_index 分组 + 拼回段落原文 |
| `src/character_analyzer.py` | narrator 硬编码 + voice_instruction 一句话描述 | 字段改为 `CharacterProfile` dataclass；`timbre` 并入 `voice_prompt`；voice_prompt 强约束「用...说」格式 |
| `src/story_director.py` | segment_directions 一一对应；fallback 兜底 | 去掉 `overall_style` / `emphasis_words` / `needs_review`；字段精简到 5 个；`pause_after_ms` → `pause_hint`（秒） |
| `src/segment_builder.py` | 把 parser + resolver 结果合并成最终 segment 的思路 | 直接由 `src_next/core/segment_builder.py` 一步做完（段落切 + 引号切） |
| `src/tts_instruction_generator.py` | — | 不参考；属于 tts 层职责 |

## 七、v1 简化实现

下列点 v1 不做，后续按需补齐：

1. **不做 overall_style**：旧 src 会先推断故事类型 / 整体基调。v1 每段独立判断。
2. **不做 emphasis_words**：旧 src 会标注重读词。v1 只有 `delivery_instruction` 一句话。
3. **不做 needs_review**：旧 src 会标记低置信度段。v1 用 `DirectorInstruction` 字段表达不出「需复核」，未来可加。
4. **不做 quote_type 规则识别**：quote_classifier 完全交给 LLM 判断，不做规则 + LLM 混合。LLM 失败 / 返回结构异常时统一 fallback 成 `unknown`（保守并回 narration）。早期版本曾尝试过正则 + LLM 混合，但中文叙事句式太多样，规则误判率高，不如纯 LLM。
5. **无 incremental 分析**：本层假设每次调用都是从头分析；不支持「已有角色档案，只增量分析新角色」。
6. **LLM 按段落分批，不做全局并行**：quote_classifier 和 story_resolver 都按段落串行调 LLM（每段一次）；character / director 各一次。未来需要加速时可以加并行。
7. **不带 debug 落盘（除 quote_classifications.json）**：旧 src 解析失败会把 raw 文本写到 `output/debug/`。v1 只有 quote_classifier 通过 `output_debug_path` 参数显式落盘 `quote_classifications.json`，其它中间产物由 core pipeline 统一负责持久化。

## 八、最小调用示例

```python
from src_next.core.data_models import StoryInput
from src_next.core.segment_builder import build_segments
from src_next.llm.mock_llm import MockLLMClient
from src_next.analysis.quote_classifier import classify_and_merge_quotes
from src_next.analysis.story_resolver import resolve_speakers
from src_next.analysis.character_analyzer import analyze_characters
from src_next.analysis.story_director import generate_director_plan

story = StoryInput(story_name="test", text="从前有一只小松鼠。\n小松鼠说：我要去找松果。")
segments = build_segments(story)

llm = MockLLMClient()  # 离线无网络也能跑通
merged = classify_and_merge_quotes(segments, llm)  # 非对白引号并回 narration
resolved = resolve_speakers(merged, llm)           # 1:1 填 speaker
characters = analyze_characters(resolved, llm)
plan = generate_director_plan(resolved, characters, llm)
```

切换到真实 Qwen 后端只需把 `MockLLMClient()` 换成 `QwenHTTPClient()`，其余代码不动。

## 九、角色归并设计

短文本（≤3000 字）中同一角色常被多个称呼拆成多个 speaker（如「小明」+「那个男孩」+「孩子」），如果不归并会导致 voicebank 重复生成 voice reference、最终音频音色不一致。本层用两层归并控制。

### 9.1 第一层：story_resolver 跨段 known_speakers

- 维护 `known_speakers: list[str]` 跨段累积
- 每段调 LLM 时把已识别角色一并传入 prompt，强约束 LLM 优先复用已知名
- LLM 返回后过 `_normalize_speaker_alias` 做**最保守归并**：仅清洗标点 + 全等命中，不做后缀剥离
- 目的：减少明显 naming drift（如不同段落 LLM 输出带不带尾标点的不一致）

### 9.2 第二层：character_analyzer LLM alias_map

- prompt 要求 LLM 输出 `characters` 数组 + `alias_map` 字典（别名 → canonical 名）
- canonical 名优先采用原文首次出现且稳定的称呼（优先级：人名 > 职务 > 代词）
- LLM 把指代同一角色的不同称呼（绰号 / 亲属称谓 / 描述性称呼）合并到 canonical
- alias_map 的 value 必须出现在 `characters[].name` 中

### 9.3 第二层 fallback：_SAFE_TITLES 安全后缀剥离

LLM 没返回 alias_map 时（结构异常 / MockLLMClient）兜底：

- 用本文件内的 `_SAFE_TITLES`（先生 / 女士 / 老师 / 师傅 / 同志 / 小姐 / 大叔 / 阿姨 / 爷爷 / 奶奶 10 个 ≥2 字、语义明确的社会称谓）
- 仅当 speaker 以这些后缀结尾，且剥离后剩下的部分（≥2 字）能命中另一个 speaker 名时才合并
- **不做前缀剥离**（老X / 小X / 阿X 风险高，留给 LLM alias_map 处理）

### 9.4 不归并的边界

- 父子关系（"小明" vs "小明爸爸"）
- 主仆关系
- 人和动物
- narrator 与故事角色（narrator 永远独立，不在 characters 数组中）

### 9.5 canonical 名选取原则

- 优先级：人名 > 职务 > 代词
- 原文首次出现且后文较稳定的称呼
- **不要凭常识改写**：原文叫"小明"就不要 canonical 改成"明"

### 9.6 已知 Gap：下游 speaker 映射未实现

本次归并结果保存在 `characters.json` 的 `CharacterProfile.aliases` 字段，但 **resolved_segments 不回写 canonical**（保持 story_resolver 1:1 in/out 契约）。意味着：

- segments 里 speaker 仍是各段 LLM 给的原始称呼（如「小明」「小明哥哥」并存）
- voicebank 按 `CharacterProfile.name`（canonical）生成 voice reference，`speaker_to_voice` 字典 key 是 canonical 名
- `tts_instruction_builder` 用 `segment.speaker`（如「小明哥哥」）查 `speaker_to_voice` 时**会查不到** → 走 fallback 失败

**后续需要做**（不在本次范围）：

- 在 `tts_instruction_builder` 或 pipeline 中段，根据 `characters[].aliases` 建立 `alias → canonical` 映射，查 `speaker_to_voice` 时先映射
- 或者在 character_analyzer 之后增加一步「segments speaker 回写 canonical」
