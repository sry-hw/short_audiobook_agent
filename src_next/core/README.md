# core/ 核心业务编排层

> 本层是 `src_next/` 重构架构中的主干。整体架构图、各层关系、数据流请见 [../README.md](../README.md) 的「〇、一图看懂 src_next」。

## 一、这一层负责什么

* 定义核心数据结构（segment、character、director plan、tts instruction 等）。
* 定义完整 pipeline 流程：txt → segments → resolved → characters → voicebank → director plan → tts instructions → audio。
* 组织 `analysis/`、`llm/`、`voicebank/`、`tts/` 四层协作。
* 保存中间产物为 JSON / 文件。
* 统计每阶段耗时和整体 RTF。
* 控制 STAGE 级别日志，让流程进度可追踪。

## 二、这一层不负责什么

* 不写具体 Gemma4 HTTP 请求细节。
* 不写具体 CosyVoice / IndexTTS / FishPro 合成细节。
* 不硬编码任何服务器地址或模型部署细节。
* 不直接做 UI 展示（交给 `app/`）。
* 不解析 yaml 配置文件格式（交给 `profiles/`）。

## 三、输入

* 原始文本（来自 `app/`）。
* Profile 配置（已加载的字典 / 对象）。
* LLM adapter 实例。
* TTS adapter 实例。
* Voicebank adapter 实例。

## 四、输出

* segments。
* resolved segments。
* character profiles。
* director plan。
* tts instructions。
* 最终音频。
* run summary（耗时、RTF、各阶段产物路径）。

## 五、未来会放的文件

```text
core/
├── data_models.py             # 核心数据结构定义
├── audiobook_pipeline.py      # 主流程编排
├── segment_builder.py         # txt → segments（段落切 + 引号切，两级）
├── tts_instruction_builder.py # segments + director_plan + voicebank → 通用 TTSInstruction
├── audio_merger.py            # audio segments → final audio
├── logging_utils.py           # STAGE 级日志、耗时统计
└── test_tts_instruction_builder.py  # builder 自检（手写输入，不依赖 LLM）
```

## 六、和其他层的交互

* **被 `app/` 调用**：接收 txt、profile、adapters。
* **调用 `analysis/`**：quote_classifier / story_resolver / character_analyzer / story_director。
* **调用 `llm/`**：通过统一接口拿 text / json（不直接发 HTTP）。
* **调用 `voicebank/`**：为角色准备音色参考。
* **调用 `tts/`**：按 tts instructions 合成音频。
* **使用 `utils/`**：JSON 保存、路径管理、音频时长读取。
* **不直接读 `profiles/` 的 yaml**：profile 由 `app/` 加载后注入。

## 七、当前实现状态（重构第 2 步）

第 2 步已经把 core 层骨架和 mock 数据流跑通。本层文件清单：

```text
core/
├── __init__.py
├── data_models.py             # ✅ 9 个 dataclass 已定义
├── segment_builder.py         # ✅ build_segments() 已实现（段落切 + 引号切，两级）
├── tts_instruction_builder.py # ✅ build_tts_instructions() 已实现（Segment + Director + Voicebank → 通用 TTSInstruction）
├── audio_merger.py            # ✅ merge_audio_segments() placeholder
├── logging_utils.py           # ✅ log_stage_start/done/item
└── audiobook_pipeline.py      # ✅ run_mock_core_pipeline() 主流程
```

### 数据结构

`data_models.py` 定义了 9 个 dataclass，覆盖链路全部中间产物：

```text
StoryInput → Segment → CharacterProfile
                    → DirectorInstruction
                    → TTSInstruction
          → VoicebankResult
          → AudioSegmentResult → AudioResult
                              → PipelineResult
```

### 入口函数

```python
from src_next.core.audiobook_pipeline import run_mock_core_pipeline

result = run_mock_core_pipeline("input/sample_story_01.txt")
# result: PipelineResult(story_name, output_dir, final_audio,
#                        success, stage_timings, artifacts, error)
```

### 输出目录

```text
output-src-next-core/<story_name>/
├── json/
│   ├── segments.json
│   ├── characters.json
│   ├── director_plan.json
│   ├── voicebank_result.json
│   ├── tts_instructions.json
│   ├── audio_result.json
│   └── pipeline_result.json
└── audio_final/
    └── <story_name>_mock.txt   # 占位文件，不生成真实 wav
```

### 哪些仍是 mock

* `characters`：只生成一个 `narrator`，未做角色抽取。
* `director_plan`：每段都是 `emotion=neutral / pace=1.0 / tone=neutral / pause_hint=0.0`，未调用 LLM。
* `voicebank_result`：`speaker_to_voice={"narrator": "mock://voice/narrator"}`，未调用任何克隆模型。
* `audio_segments`：每段 audio_path 都是 `mock://audio/<seg_id>`，未生成真实音频。
* `merge_audio_segments`：只汇总路径和 success 状态，未真正拼接 wav。

### 不在本步范围

* 不接真实 LLM（属于 `llm/` 层）。
* 不接真实 TTS（属于 `tts/` 层）。
* 不做对话者识别 / 角色抽取 / 导演计划生成（属于 `analysis/` 层）。
* 不准备真实音色（属于 `voicebank/` 层）。
* 不解析 profile yaml（属于 `profiles/` 层）。

