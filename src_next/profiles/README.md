# profiles/ 运行配置层

> 整体架构请见 [../README.md](../README.md) 的「〇、一图看懂 src_next」。本层负责描述运行时使用哪种 LLM / voicebank / TTS backend、输出位置和 pipeline 开关。

## 一、这一层负责什么

* 用 yaml 描述当前运行使用哪种 LLM / voicebank / TTS backend。
* 描述输出目录、并发参数、超时等运行时选项。
* 支持服务器 / 蓝区本地 / Mock 环境之间通过配置切换，**不改代码**。

## 二、这一层不负责什么

* 不包含业务逻辑（不知道什么是角色、导演计划）。
* 不调用模型、不发 HTTP 请求。
* 不实现 adapter（adapter 在 `llm/`、`tts/`、`voicebank/` 里）。
* 不参与 pipeline 编排。

## 三、Profile 分两类

### 1. 单模块 profile（adapter 单测用）

只包含一个模块的配置，给 adapter 专项测试脚本使用。

| 文件 | 包含块 | 用途 |
|---|---|---|
| `blue_indextts.yaml` | `tts` | `test_tts_from_artifacts.py` 读它调 IndexTTS adapter |
| `blue_qwen_voicegenerator.yaml` | `voicebank` | voicebank 单测（蓝区 WSL 调 Qwen VoiceDesign） |
| `server_qwen_voicegenerator.yaml` | `voicebank` | voicebank 单测（服务器路径占位） |

调用方式（单模块测试）：

```bash
python -m src_next.tts.test_tts_from_artifacts \
    --artifact-dir output-src-next-analysis-test/桂花雨 \
    --profile src_next/profiles/blue_indextts.yaml
```

### 2. 完整 pipeline profile（端到端链路用）

同时包含 `llm` / `voicebank` / `tts` / `output` / `pipeline` 五块配置，给未来的 `core/audiobook_pipeline.py` 真实链路使用。

**命名格式**：

```text
<region>_<voicebank>_<tts>[_mode].yaml
```

| 字段 | 含义 |
|---|---|
| `region` | `blue`（蓝区）/ `server`（服务器）/ `mock`（离线） |
| `voicebank` | `qwenvoice` / `cosyvoice` / `mockvoice` ... |
| `tts` | `indextts` / `cosyvoice` / `fishpro` / `qwen_tts` / `mocktts` ... |
| `mode`（可选） | `batch` / `single` / `streaming`，不写表示默认 |

**当前已有**：

| 文件 | 组合 |
|---|---|
| `blue_qwenvoice_indextts_batch.yaml` | 蓝区 Qwen VoiceDesign voicebank + IndexTTS batch 合成 |

调用方式（未来 pipeline）：

```bash
python -m src_next.core.audiobook_pipeline \
    --input input/桂花雨.txt \
    --profile src_next/profiles/blue_qwenvoice_indextts_batch.yaml
```

## 四、完整 pipeline profile 必填块

每个完整 pipeline profile **必须**包含以下 5 个顶层块：

```yaml
name: <profile_name>          # 顶层标识，便于日志引用
region: <blue|server|mock>
description: ...              # 可选，多行说明

llm:
  backend: ...                # 必填，对应 llm/registry 支持的 backend
  ...

voicebank:
  backend: ...                # 必填
  ...

tts:
  backend: ...                # 必填
  ...

output:
  root: ...                   # 必填，pipeline 输出根目录

pipeline:
  save_intermediate_json: ...
  reuse_existing: ...
  stop_on_tts_error: ...
```

**关键约定**：

* **文件名只用于人工识别组合**；程序逻辑以 yaml 内容为准。
* **pipeline 统一加载一个完整 profile**，不自行拼接多个文件。
* **`llm` / `voicebank` / `tts` 三个 adapter 层不自行寻找 profile 文件**——它们由 pipeline 注入对应配置块。
* pipeline 读取完整 yaml 后，把 `llm` / `voicebank` / `tts` 三块配置分别传给对应 registry（`create_llm_client` / `create_voicebank_adapter` / `create_tts_adapter`）。

## 五、完整 pipeline profile 示例

```yaml
name: blue_qwenvoice_indextts_batch
region: blue
description: 蓝区完整链路：Qwen3.6-plus + Qwen VoiceDesign + IndexTTS batch

llm:
  backend: qwen_http
  model: qwen3.6-plus
  base_url_env: QWEN_BASE_URL
  api_key_env: QWEN_API_KEY
  timeout: 300

voicebank:
  backend: qwen_voicegenerator
  generator_root: "F:/akoasm/short_audiobook_agent/src_next/voicebank/scripts"
  script_path: "run_voicedesign_srcnext.py"
  model_path: "F:/akoasm/TTS-test/models/qwen3-tts-voicedesign"
  python_executable:
    - "wsl"
    - "/mnt/f/akoasm/TTS-test/envs/qwen3-customvoice/bin/python"
  output_subdir: "voicebank"
  extra_args:
    device: "cuda:0"

tts:
  backend: indextts
  engine_root: "F:/akoasm/TTS-test/engines/indextts"
  batch_wrapper_path: "F:/akoasm/short_audiobook_agent/src_next/tts/scripts/run_indextts_batch.py"
  python_executable: "F:/akoasm/venv-indextts/python.exe"
  output_subdir: "audio_segments"

output:
  root: "output-src-next-pipeline"

pipeline:
  save_intermediate_json: true
  reuse_existing: false
  stop_on_tts_error: false
```

## 六、运行命令约定

### 默认（最常用）

```bash
python -m src_next.core.audiobook_pipeline \
    --input input/sample_story_01.txt \
    --profile src_next/profiles/blue_qwenvoice_indextts_batch.yaml
```

| 参数 | 必填 | 默认值 |
|---|---|---|
| `--input` | ✅ | — |
| `--profile` | ✅ | — |
| `--output-root` | ❌ | profile 的 `output.root`（即 `output-src-next-pipeline`） |
| `--story-name` | ❌ | input 文件名 stem（如 `sample_story_01`） |
| `--mock` | ❌ | 走 `run_mock_core_pipeline`，忽略 `--profile` |
| `--reuse-existing` | ❌ | 强制 `pipeline.reuse_existing=true`（覆盖 profile） |

### 覆盖输出位置 / 故事名

```bash
python -m src_next.core.audiobook_pipeline \
    --input input/sample_story_01.txt \
    --profile src_next/profiles/blue_qwenvoice_indextts_batch.yaml \
    --output-root output-custom \
    --story-name sample_story_01_v2
```

### 长文本回归（可选）

```bash
python -m src_next.core.audiobook_pipeline \
    --input input/桂花雨.txt \
    --profile src_next/profiles/blue_qwenvoice_indextts_batch.yaml
```

### Mock 回归（不需要 profile）

```bash
python -m src_next.core.audiobook_pipeline \
    --mock \
    --input input/sample_story_01.txt
```

## 七、和其他层的交互

* **被 `core/audiobook_pipeline.py` 读取**：pipeline 解析 yaml，把 `llm` / `voicebank` / `tts` 三块分别传给对应 registry 实例化 adapter。
* **决定 `llm/`、`tts/`、`voicebank/` 的具体后端**：通过 `backend` 字段切换。
* **被 `core/` 间接使用**：core 拿到的是已实例化的 adapter，不直接读 yaml。
* **不依赖任何业务层**：本层是纯配置描述。

## 八、新增 profile 的检查清单

提交新 profile 前确认：

1. 单模块 profile：只包含一个模块块（`tts` 或 `voicebank`）。
2. 完整 pipeline profile：
   * 文件名符合 `<region>_<voicebank>_<tts>[_mode].yaml`。
   * 5 个块齐全：`llm` / `voicebank` / `tts` / `output` / `pipeline`。
   * 路径复用已经跑通的单模块 profile，不发明新路径。
   * 必要字段（如 `python_executable`、`engine_root`）非空。
3. yaml 可被 `yaml.safe_load` 正常解析。
