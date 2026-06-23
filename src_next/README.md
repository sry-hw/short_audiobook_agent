# src_next 重构说明

`src_next/` 是有声书 Agent 项目的下一代重构目录。它与旧目录 `src/` 并列存在，用于在不破坏旧链路的前提下，逐步搭建更清晰、更可维护、更容易切换模型后端的新架构。

旧 `src/` 中已经有可以运行的链路，但随着项目逐渐支持多种 LLM、多种 TTS、多种音色克隆模型，以及服务器 / 蓝区本地等不同运行环境，原有结构中“业务流程”和“具体模型后端”耦合较强，继续扩展会变得混乱。因此，`src_next/` 的目标是重新划分层次，把有声书生成流程拆成清晰的业务编排层、语义分析层、LLM 适配层、TTS 适配层、音色克隆适配层和运行配置层。

## 〇、一图看懂 src_next

如果你是第一次接触这个项目，先看下面这两张图，就能大致理解整体结构和每一层的职责。后面的章节是对这两张图的展开。

### 1. 分层架构总览

```text
txt + profile
     │
     ▼
app/                            ← 应用入口层
     │                            启动 / 读 profile / 调用主流程
     ▼
core/                           ← 核心业务编排层
     │                            组织各层协作，串起完整生成链路
     │
     │   core/ 调用以下四个适配 / 分析层
     │   (具体使用哪个后端，由 profiles/ 决定):
     │
     ├──► analysis/   语义分析       (对话者识别 / 角色抽取 / 导演计划)
     ├──► llm/        大模型适配     (Gemma4 HTTP / Mock / 本地模型)
     ├──► voicebank/  音色库适配     (CosyVoice / IndexTTS / FishPro)
     └──► tts/        语音合成适配   (CosyVoice / IndexTTS / FishPro / 本地 TTS)
                              │
                              ▼
                          最终音频 (mp3 / wav)


底层支撑 (所有层共用):
  profiles/    运行配置层 ── yaml 描述使用哪种 LLM / TTS / voicebank 后端
  utils/       通用工具层 ── io / json / audio / 路径工具
```

**关键理解**:

- 业务流程 (`core/`) **不直接依赖**具体模型，只调用统一接口。
- 每个适配层 (`llm/`、`tts/`、`voicebank/`) 都**可插拔**，由 `profiles/` 配置决定具体后端。
- `analysis/` 是纯业务层，调用 `llm/` 完成故事理解，但**不知道** TTS 部署细节。
- 切换服务器 / 蓝区本地 / Mock 环境 = 换一个 profile，**不改代码**。

### 2. 各层职责速查表

| 层级 | 主要职责 | 输入 → 输出 | 不应该知道 |
|------|---------|------------|-----------|
| `app/` | 启动入口、读取配置 | txt 路径 + profile 名 → 启动 pipeline | 具体模型的请求格式 |
| `core/` | 组织流程、保存中间产物、统计耗时 | 原始文本 + adapters → 有声书成品 + JSON | Gemma4 / CosyVoice 调用细节 |
| `analysis/` | 对话者识别、角色抽取、导演计划 | segments + llm → resolved / characters / director plan | TTS 部署地址、音色克隆细节、**TTS 后端专用参数**（director_plan 是通用语义层，不写 IndexTTS / CosyVoice 专用字段） |
| `llm/` | 统一 LLM 接口、多后端切换、JSON 解析重试 | prompt → text / json | 故事业务、TTS 是什么 |
| `tts/` | 合成音频、多后端切换 | tts instructions + voicebank → audio segments | 角色分析逻辑、LLM 调用细节 |
| `voicebank/` | 准备角色音色参考、多后端切换 | character profiles → voice reference | 正文音频合成、故事分析 |
| `profiles/` | 描述运行环境、切换模型组合 | yaml 配置 → 选定 adapter 组合 | 具体业务逻辑 |
| `utils/` | 文件 / JSON / 音频 / 路径工具 | 通用输入 → 通用输出 | 当前是哪条 pipeline |

> 经验法则：只要每一层都能清楚回答「**接收什么 / 输出什么 / 不应该知道什么**」这三个问题，结构就能保持稳定。

## 一、重构目标

`src_next/` 的目标不是简单复制旧代码，而是建立一套更清楚的工程结构，使项目具备以下能力：

1. 同一条有声书生成链路可以切换不同 LLM，例如 Gemma4、蓝区本地模型、Mock LLM。
2. 同一条链路可以切换不同 TTS 或音色克隆模型，例如 CosyVoice、IndexTTS、FishPro、本地轻量 TTS。
3. 蓝区本地环境和服务器环境通过配置切换，而不是通过手动修改代码切换。
4. 每一层代码职责清晰，输入输出明确，便于理解、调试、替换和扩展。
5. 在重构过程中保留旧 `src/`，保证旧链路不被破坏。
6. 先用 Mock 链路跑通结构，再逐步接入真实模型。

## 二、核心设计原则

### 1. 旧链路不动，新链路并行构建

`src_next/` 是新的实验性重构目录。任何重构工作都不应该直接修改旧 `src/` 的稳定链路。旧代码可以被阅读、引用和适配，但不能在未确认的情况下大规模改动。

### 2. 业务流程不直接依赖具体模型

主流程不应该直接调用某个具体模型，例如：

```python
from src.cosyvoice.tts_engine import synthesize_all
from src.gemma4.llm_client import call_gemma4_json
```

而应该依赖统一接口，例如：

```python
llm_client.generate_json(prompt)
tts_adapter.synthesize_segments(instructions, voicebank, output_dir)
voicebank_adapter.prepare_voicebank(characters, output_dir)
```

具体使用 Gemma4、CosyVoice、IndexTTS 还是蓝区本地模型，由配置决定。

### 3. 任何真实模型都必须经过 adapter

LLM、TTS、Voicebank 都需要有统一适配层。业务代码不直接知道模型部署在哪里，也不直接关心模型调用细节。

### 4. 每一步都要有明确输入和输出

有声书 Agent 的数据流必须可追踪：

```text
txt 原始文本
  → segments
  → speaker-resolved segments
  → character profiles
  → director plan
  → tts instructions
  → voicebank
  → audio segments
  → final audio
```

每个中间产物都应可以保存成 JSON 或文件，便于调试和复现。

### 5. 优先 Mock 可运行，再接真实模型

`src_next/` 初期应先构建 Mock 链路。Mock 链路不调用真实 LLM 和真实 TTS，但能完整跑通数据流。这有助于理解结构，也能在蓝区本地无服务器时继续开发。

## 三、推荐目录结构

```text
src_next/
├── README.md
├── app/
├── core/
├── analysis/
├── llm/
├── tts/
├── voicebank/
├── profiles/
└── utils/
```

未来完整结构可以进一步扩展为：

```text
src_next/
├── README.md
│
├── app/
│   ├── run.py
│   ├── run_mock.py
│   ├── pipeline.py
│   └── webui.py
│
├── core/
│   ├── data_models.py
│   ├── audiobook_pipeline.py
│   ├── segment_builder.py
│   ├── tts_instruction_builder.py
│   ├── audio_merger.py
│   └── logging_utils.py
│
├── analysis/
│   ├── story_resolver.py
│   ├── character_analyzer.py
│   └── story_director.py
│
├── llm/
│   ├── base.py
│   ├── mock_llm.py
│   ├── gemma4_http.py
│   ├── local_llm.py
│   └── parallel.py
│
├── tts/
│   ├── base.py
│   ├── mock_tts.py
│   ├── cosyvoice_adapter.py
│   ├── indextts_adapter.py
│   ├── fishpro_adapter.py
│   └── local_tts.py
│
├── voicebank/
│   ├── base.py
│   ├── mock_voicebank.py
│   ├── cosyvoice_voicebank.py
│   ├── indextts_voicebank.py
│   └── fishpro_voicebank.py
│
├── profiles/
│   ├── mock_debug.yaml
│   ├── server_gemma4_cosy.yaml
│   ├── server_gemma4_index.yaml
│   └── blue_local_light.yaml
│
└── utils/
    ├── io_utils.py
    ├── json_utils.py
    └── audio_utils.py
```

## 四、各层职责说明

## 1. app 层：应用入口层

`app/` 负责用户如何启动系统。它不负责具体模型逻辑，也不直接实现文本分析或音频合成。

### 主要职责

* 提供命令行入口。
* 提供 WebUI pipeline 入口。
* 读取 profile 配置。
* 调用 core 层的主流程。
* 负责把用户输入文件传入主链路，并展示最终结果。

### 典型文件

```text
app/run.py
app/run_mock.py
app/pipeline.py
app/webui.py
```

### 输入

* 用户输入的 txt 文件路径。
* 用户选择的运行 profile。
* WebUI 输入文本或上传文件。

### 输出

* 控制台日志。
* 最终生成结果路径。
* WebUI 可展示的 result 字典。

### 不应该负责

* 不直接调用 LLM。
* 不直接调用 TTS。
* 不直接解析故事结构。
* 不直接拼接音频。

## 2. core 层：核心业务流程层

`core/` 是有声书 Agent 的主干。它负责组织“文本如何一步步变成音频”的整体流程。

### 主要职责

* 定义核心数据结构。
* 定义完整 pipeline 流程。
* 组织 analysis、llm、voicebank、tts 等层协作。
* 保存中间产物。
* 统计耗时和 RTF。
* 控制 STAGE 级别日志。

### 典型文件

```text
core/data_models.py
core/audiobook_pipeline.py
core/segment_builder.py
core/tts_instruction_builder.py
core/audio_merger.py
core/logging_utils.py
```

### 输入

* 原始文本。
* Profile 配置。
* LLM adapter。
* TTS adapter。
* Voicebank adapter。

### 输出

* segments。
* characters。
* director plan。
* tts instructions。
* audio result。
* run summary。

### 不应该负责

* 不写具体 Gemma4 HTTP 请求。
* 不写具体 CosyVoice 合成细节。
* 不写具体 IndexTTS 推理细节。

## 3. analysis 层：语义分析层

`analysis/` 负责把故事文本理解成适合有声书生成的结构化信息。

### 主要职责

* 对话者识别。
* 引号归属判断。
* 角色抽取。
* 角色声音分析。
* 导演计划生成。
* 把 LLM 输出转成稳定结构。

### 典型文件

```text
analysis/story_resolver.py
analysis/character_analyzer.py
analysis/story_director.py
```

### 输入

* segments。
* LLM client。
* 上下文信息。
* 角色列表。

### 输出

* speaker-resolved segments。
* character profiles。
* director plan。

### 不应该负责

* 不直接写 requests.post。
* 不关心 Gemma4 或本地模型部署细节。
* 不生成音频。
* 不准备 voicebank。

## 4. llm 层：LLM 适配层

`llm/` 负责把不同 LLM 后端封装成统一接口。

### 主要职责

* 封装不同 LLM 的调用方式。
* 提供统一的 `generate_text` 和 `generate_json` 接口。
* 处理 JSON 解析、重试、错误兜底。
* 提供并行 batch 调用工具。
* 支持服务器模型、蓝区本地模型和 mock 模型切换。

### 典型文件

```text
llm/base.py
llm/mock_llm.py
llm/gemma4_http.py
llm/local_llm.py
llm/parallel.py
```

### 输入

* Prompt。
* LLM 配置。
* batch 任务。

### 输出

* 文本结果。
* JSON 结果。
* batch 并行结果。

### 不应该负责

* 不理解故事业务。
* 不知道什么是 TTS。
* 不生成 director plan 的业务结构，只返回模型结果。

## 5. tts 层：TTS 合成适配层

`tts/` 负责把不同 TTS 后端封装成统一接口。

### 主要职责

* 接收通用 `TTSInstruction`（模型无关，由 `core/tts_instruction_builder` 产出）。
* 调用具体 TTS 模型或服务（IndexTTS / CosyVoice / FishPro / Qwen TTS / mock）。
* 生成每个 segment 的 wav 文件。
* 单条失败隔离（失败 instruction 写到 `AudioSegmentResult.error`，不阻断其他条）。
* 缓存（已存在 wav 可复用）+ dry_run 模式。

### 典型文件

```text
tts/base.py                  # TTSError + BaseTTSAdapter(synthesize)
tts/mock_tts.py              # ✅ 离线占位
tts/indextts_adapter.py      # ✅ subprocess 调外部 IndexTTS CLI
tts/cosyvoice_adapter.py     # 🚧 占位
tts/fishpro_adapter.py       # 🚧 占位
tts/qwen_tts_adapter.py      # 🚧 占位
tts/registry.py              # create_tts_adapter(backend, **config)
tts/test_tts_from_artifacts.py  # 不重跑 analysis 的 smoke test
```

详细接口约定、IndexTTS 真实参数、测试方法请见 [`tts/README.md`](tts/README.md)。

### 输入

* `list[TTSInstruction]`（通用合成指令）。
* `VoicebankResult`（speaker → voice_ref 映射，用于 fallback 和审计）。
* `output_dir`（adapter 在其下创建 `output_subdir` 放 wav + log）。
* `dry_run` / `limit` / `timeout_per_seg` 等 kwargs。

### 输出

* `list[AudioSegmentResult]`（与 instructions 一一对应，按顺序）。
* per-segment log + adapter 配置快照（落到 `output_dir/<output_subdir>/`）。

### 不应该负责

* 不切分文本（segment_builder）。
* 不识别说话人（story_resolver）。
* 不抽取角色 / 不生成导演计划（character_analyzer / story_director）。
* 不准备音色参考（voicebank）。
* 不拼接多段 wav（audio_merger）。

### 关于 IndexTTS 的特殊性

IndexTTS 是纯 zero-shot voice cloning，**不支持** pace / emotion / volume /
pitch / stress_words / delivery_instruction 等风格参数。这些通用字段在
`TTSInstruction` 里保留是为了让 CosyVoice2 / Qwen3-TTS 等支持风格控制的后端
能用；IndexTTS adapter 会把它们写到 per-segment log 留档，但**不影响合成结果**。

## 6. voicebank 层：音色克隆 / 音色参考适配层

`voicebank/` 负责准备角色声音参考。它和 `tts/` 分开，是因为有些 TTS 后端需要提前生成 voicebank，有些 TTS 后端不需要。

### 主要职责

* 根据 character profiles 生成或查找角色音色参考。
* 支持不同音色克隆模型。
* 管理 narrator 和角色 voice reference。
* 返回 TTS 可使用的 voicebank 结果。

### 典型文件

```text
voicebank/base.py
voicebank/mock_voicebank.py
voicebank/cosyvoice_voicebank.py
voicebank/indextts_voicebank.py
voicebank/fishpro_voicebank.py
```

### 输入

* character profiles。
* 输出目录。
* voicebank backend 配置。

### 输出

* voicebank result。
* 每个 speaker 对应的 voice reference 路径或 id。

### 不应该负责

* 不合成正文音频。
* 不生成 TTS 指令。
* 不调用故事分析 LLM。

## 7. profiles 层：运行配置层

`profiles/` 用于描述不同运行环境和模型组合。

### 主要职责

* 定义当前使用哪种 LLM。
* 定义当前使用哪种 voicebank。
* 定义当前使用哪种 TTS。
* 定义输出目录。
* 定义并发、batch size、timeout 等参数。
* 支持服务器环境和蓝区本地环境切换。

### 典型配置

```text
profiles/mock_debug.yaml
profiles/server_gemma4_cosy.yaml
profiles/server_gemma4_index.yaml
profiles/blue_local_light.yaml
```

### 示例

```yaml
profile_name: server_gemma4_cosy

llm:
  backend: gemma4_http
  base_url: http://10.154.39.83:8000/v1/chat/completions
  model: gemma4-31B
  max_workers: 3
  bypass_proxy: true

voicebank:
  backend: cosyvoice
  output_subdir: voicebank

tts:
  backend: cosyvoice
  output_subdir: audio_segments

output:
  root: output-src-next
```

## 8. utils 层：通用工具层

`utils/` 放置与业务无关的通用工具。

### 主要职责

* 文件读写。
* JSON 保存和加载。
* 路径管理。
* 音频时长读取。
* 文本清理。
* 安全字符串截断。

### 典型文件

```text
utils/io_utils.py
utils/json_utils.py
utils/audio_utils.py
```

### 不应该负责

* 不包含具体业务逻辑。
* 不直接调用模型。
* 不知道当前是哪个 pipeline。

## 五、完整数据流

理想情况下，一个 txt 文本会按如下流程处理：

### 1. 数据流图示

```text
   ┌──────────┐
   │ 原始 txt │
   └────┬─────┘
        │  ①  core/segment_builder.py
        ▼
   ┌──────────────────┐
   │    segments      │  分段（段落 + 引号切，所有引号先变 dialogue candidate）
   └────┬─────────────┘
        │  ②  analysis/quote_classifier.py  + llm
        ▼
   ┌──────────────────────┐
   │  merged segments     │  非对白引号（书名/强调词）并回 narration
   └────┬─────────────────┘
        │  ③  analysis/story_resolver.py  + llm
        ▼
   ┌──────────────────────┐
   │  resolved segments   │  识别每段是谁在说（1:1）
   └────┬─────────────────┘
        │  ④  analysis/character_analyzer.py  + llm
        ▼
   ┌──────────────────────┐
   │  character profiles  │  角色档案（年龄 / 性格 / 音色特征）
   └────┬─────────────────┘
        │  ⑤  voicebank/  (为每个角色准备音色参考)
        ▼
   ┌──────────┐
   │ voicebank│  每个角色对应的音色参考
   └────┬─────┘
        │  ⑥  analysis/story_director.py  + llm
        ▼
   ┌───────────────┐
   │ director plan │  通用语义导演层（emotion / intensity / pace / tone /
   │               │                 volume / pitch / pause_hint / stress_words
   │               │                 / delivery_instruction，不绑 TTS 后端）
   └────┬──────────┘
        │  ⑦  core/tts_instruction_builder.py
        ▼
   ┌───────────────────┐
   │  tts instructions │  **模型无关的通用合成指令**（text + voice_ref +
   │                   │   emotion + intensity + pace + tone + volume +
   │                   │   pitch + pause_hint + stress_words +
   │                   │   delivery_instruction + output_filename + metadata）
   │                   │   不写 IndexTTS / CosyVoice / FishPro / Qwen TTS
   │                   │   专用字段；各 TTS adapter 负责翻译。
   └────┬──────────────┘
        │  ⑧  tts/  (按指令合成音频)
        ▼
   ┌────────────────┐
   │ audio segments │  每段对应的音频片段
   └────┬───────────┘
        │  ⑨  core/audio_merger.py
        ▼
   ┌──────────────┐
   │   最终音频   │  成品有声书
   └──────────────┘
```

**一眼看懂**：txt 经过 9 个阶段变成有声书，**①⑦⑨ 在 `core/`**，**②③④⑥ 在 `analysis/`**，**⑧ 在 `tts/`**；其中 ②③④⑥ 都需要调用 LLM，⑤⑧ 都依赖 voicebank 结果。每个中间产物都应能保存成 JSON 或文件，方便调试和复现。

### 2. 详细步骤说明

```text
1. app/run.py
   接收 txt 路径和 profile 名称

2. profiles
   加载运行配置，决定使用哪个 LLM、哪个 TTS、哪个 voicebank

3. core/audiobook_pipeline.py
   创建 pipeline，上下文和输出目录

4. core/segment_builder.py
   原始 txt → segments（段落 + 引号切，所有引号先变 dialogue candidate）

5. analysis/quote_classifier.py
   segments + llm_client → merged segments（非对白引号并回 narration）

6. analysis/story_resolver.py
   merged segments + llm_client → speaker-resolved segments（1:1）

7. analysis/character_analyzer.py
   resolved segments + llm_client → character profiles

8. voicebank adapter
   character profiles → voicebank result

9. analysis/story_director.py
   resolved segments + character profiles + llm_client → director plan

10. core/tts_instruction_builder.py
    segments + characters + director plan + voicebank result → tts instructions

11. tts adapter
    tts instructions + voicebank result → audio segments

12. core/audio_merger.py
    audio segments → final audio

13. core/audiobook_pipeline.py
    汇总 final_audio、timings、RTF、run_summary
```

## 六、中间产物

每次运行应尽量输出以下中间产物，便于调试：

```text
output-src-next/<story_name>/
├── input.txt
├── json/
│   ├── segments_raw.json              ← build_segments 后（含所有 dialogue candidate）
│   ├── quote_classifications.json     ← quote_classifier 的 LLM 判断结果
│   ├── segments_after_quote_merge.json ← 非对白引号并回 narration 后
│   ├── resolved_segments.json
│   ├── characters.json
│   ├── voicebank_result.json
│   ├── director_plan.json
│   ├── tts_instructions.json
│   ├── audio_segment_results.json     ← tts adapter 产出（每段合成结果）
│   └── run_summary.json
├── voicebank/                          ← 每 speaker 一个 wav + log
├── audio_segments/                     ← tts adapter 落盘的每段 wav + log
└── audio_final/                        ← 拼接后的最终 wav
```

## 七、Mock 链路的意义

Mock 链路是 `src_next/` 初期最重要的验证工具。

它不依赖服务器，不依赖真实 LLM，不依赖真实 TTS，可以在蓝区本地完整跑通：

```text
txt → mock segments → mock characters → mock director plan → mock tts instructions → mock audio result
```

Mock 链路的目标不是效果真实，而是验证结构正确：

* 每一层能否正常接收输入；
* 每一层能否正常输出；
* 中间 JSON 是否保存；
* pipeline 是否能跑完；
* 目录结构是否清晰；
* 日志是否可读。

## 八、后续真实模型接入顺序

推荐接入顺序如下：

1. 先完成 `src_next/` 目录骨架和 README。
2. 完成 core 数据结构和 mock pipeline。
3. 完成 mock LLM、mock voicebank、mock TTS。
4. 跑通 `app/run_mock.py`。
5. 接入 Gemma4 HTTP LLM。
6. 接入 CosyVoice 或本地 TTS adapter。
7. 接入 IndexTTS / FishPro adapter。
8. 最后接 WebUI。

## 九、开发约束

后续让 Claude 修改代码时，需要遵守：

1. 每次只改一个阶段。
2. 每次改动前先说明要改哪些文件。
3. 每次改动后运行 `py_compile`。
4. 每次完成后输出本阶段输入、输出、职责和验证结果。
5. 不要修改旧 `src/`，除非明确要求。
6. 不要把模型调用写进业务代码。
7. 不要把服务器地址硬编码进 analysis 层或 core 层。
8. 不要在没有确认的情况下接 WebUI。
9. 不要一次性重构全部项目。
10. 每个 adapter 必须说明它接收什么、返回什么。

## 十、Owner 需要重点关注的问题

作为项目 owner，需要持续检查每一层是否回答清楚以下三个问题：

```text
这一层接收什么？
这一层输出什么？
这一层不应该知道什么？
```

例如：

* analysis 层可以知道故事和角色，但不应该知道 TTS 模型部署在哪里。
* TTS 层可以知道语音合成指令，但不应该知道角色是如何由 LLM 分析出来的。
* app 层可以知道用户选择了哪个 profile，但不应该知道具体模型请求格式。
* core 层可以组织流程，但不应该硬编码某个模型后端。

只要这三个问题清楚，项目结构就能保持稳定、可维护和可扩展。
