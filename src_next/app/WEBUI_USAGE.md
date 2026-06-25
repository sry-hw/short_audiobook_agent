# src_next 有声书 WebUI 使用说明

> 本文档面向两类读者：
> 1. **使用者** — 知道如何打开网页、选择 profile、输入文本、下载结果；
> 2. **开发 / 排障者** — 知道如何启动服务、检查状态、查看日志、关闭服务、定位任务输出。
>
> 所有命令、字段、路径均以 `src_next/app/gradio_webui.py` / `src_next/core/audiobook_pipeline.py` / `src_next/core/logging_utils.py` 当前代码为准。

---

## 1. WebUI 功能概述

本 WebUI 是基于 **`src_next` 新链路**的有声书生成界面，**不再使用** 旧 `src/` 链路（旧 `webui_old.py` 已废弃，不在本文档范围内）。

支持的功能：

- 文本框直接粘贴故事文本；
- 上传 TXT 文件（UTF-8 / GBK 自动识别）；
- 下拉选择 YAML profile（决定 LLM / voicebank / TTS 组合）；
- 实时显示 10 个阶段的进度面板；
- 实时显示运行日志；
- 生成最终音频后展示播放器和下载按钮；
- 显示 task_id、任务目录、总耗时、音频时长、RTF；
- 每次运行写入完整任务日志到 `<task_dir>/logs/pipeline.log`。

---

## 2. 启动方式

### 前台启动（调试用）

```bash
python -m src_next.app.gradio_webui \
  --host 0.0.0.0 \
  --port 7860 \
  --concurrency 5 \
  --queue-size 20
```

### 后台启动（服务器常驻）

```bash
nohup python -m src_next.app.gradio_webui \
  --host 0.0.0.0 \
  --port 7860 \
  --concurrency 5 \
  --queue-size 20 \
  > webui.log 2>&1 &
```

### 参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址；`0.0.0.0` 表示所有网卡 |
| `--port` | `7860` | 监听端口 |
| `--concurrency` | `5` | **同时运行的生成任务**上限 |
| `--queue-size` | `20` | 排队任务上限 |
| `--share` | False | 是否创建 Gradio 公开链接（一般不用） |

> **关于并发**：`--concurrency 5` 指**最多 5 个生成任务同时运行**。打开网页本身**不**占用生成并发；只有点击「生成有声书」才会进入队列。

Gradio 3.x 和 4.x 的 `queue()` API 不同，本 WebUI 自动尝试两种签名；服务器上无论装哪个版本都能跑。

---

## 3. 访问方式

### 黄区浏览器

```text
http://10.50.121.102:7860
```

### 蓝区

通常**无法**直接访问该内网链接，主要在黄区浏览器查看。

---

## 4. 检查 WebUI 是否启动成功

### 检查进程

```bash
ps aux | grep gradio_webui
```

### 查看启动日志

```bash
tail -f webui.log
```

### 检查端口监听

```bash
lsof -i:7860
```

服务器若没有 `lsof`，可改用：

```bash
netstat -tulnp | grep 7860
```

### 本机 HTTP 探活

```bash
curl -I http://127.0.0.1:7860
```

应返回 `HTTP/1.0 200 OK` 或 `HTTP/1.1 200 OK`。

### 黄区浏览器最终确认

打开 `http://10.50.121.102:7860`，应看到「有声书生成器」标题和左侧文本框 / 右侧 profile 下拉框。

---

## 5. 关闭 WebUI

### 找 PID

```bash
ps aux | grep gradio_webui
```

输出第二列就是 PID。

### 正常关闭

```bash
kill <PID>
```

### 强制关闭

```bash
kill -9 <PID>
```

### 确认端口释放

```bash
lsof -i:7860
```

或：

```bash
netstat -tulnp | grep 7860
```

无输出表示端口已释放。

---

## 6. 页面输入说明

### 两种输入方式

1. **文本框直接粘贴**：在左侧「故事文本」框中粘贴故事内容；
2. **上传 TXT 文件**：点击「或上传 TXT 文件」选择本地 `.txt` 文件。

上传成功后，文件内容会自动填入文本框；可继续在文本框中编辑。

### 输入限制

| 限制项 | 值 | 触发时机 | 错误提示 |
|---|---|---|---|
| 文本长度上限 | **3500 字** | 点击「生成」 / 实时字数统计 | `文本超出 3500 字限制（当前 N 字）` |
| TXT 文件大小上限 | **20 KB**（20480 字节） | 文件上传时 | `文件过大（N KB），最大 20 KB；请缩短或拆分` |
| TXT 内容字符上限 | 3500 字 | 文件读取后二次校验 | `文件内容超出 3500 字限制（N 字），请缩短` |
| 文本为空 | — | 点击「生成」 | `请输入故事文本` |
| 未选 profile | — | 点击「生成」 | `请先选择一个 profile` |

> **重要**：超出任何限制时，**网页端会立即显示 UX 错误提示并恢复生成按钮**，不会进入生成流程。

### TXT 编码读取逻辑

读取顺序：

1. UTF-8（带或不带 BOM）；
2. UTF-8 失败 → 尝试 GBK；
3. 仍失败 → 显示 `文件读取失败：未知编码（请用 UTF-8 或 GBK 保存）`。

其他编码（UTF-16 / UTF-32 / 二进制）不支持。

---

## 7. Profile 选择说明

### Profile 是什么

一个 YAML profile 决定本次生成使用的：

- **LLM** 后端（如 Gemma4 HTTP / Qwen HTTP）；
- **Voicebank** 后端（如 Qwen3 VoiceDesign HTTP）；
- **TTS** 后端（如 IndexTTS HTTP / CosyVoice3 HTTP）；
- **Pipeline 开关**（`save_intermediate_json` / `reuse_existing` / `stop_on_tts_error`）。

> **注**：profile 里的 `output.root` 字段 **WebUI 会忽略**——WebUI 强制用自己的根目录 `output-src-next-webui/<profile_stem>/`，避免和 CLI 跑的 `output-src-next/<story_name>/` 产物混在一起（详见第 10 节）。

### 下拉框可能的组合（示例）

实际显示取决于 `src_next/profiles/` 下有哪些完整 profile。当前可能出现的组合类型：

```text
Gemma4 + Qwen3 VoiceDesign + IndexTTS
Gemma4 + Qwen3 VoiceDesign + CosyVoice3
```

切换 profile 后，下方「Profile 描述」会显示该 profile 的友好名 + region + description。

### Profile 来源（自动发现）

WebUI 启动时扫描 `src_next/profiles/*.yaml`，并按以下规则过滤：

1. **蓝区过滤（强制）**：`filename_stem` 以 `blue_` 开头，或 `region: blue` 的 yaml **一律不显示**。本服务部署在黄区内网，蓝区 profile（依赖本机 GPU / subprocess）跑不起来；
2. **缺块过滤**：YAML 必须包含完整 5 个顶层块：`llm` / `voicebank` / `tts` / `output` / `pipeline`。任一块缺失则**不显示**（如只配 `tts` 单块的 `blue_indextts.yaml` 双重命中：蓝区 + 缺块）；
3. **webui.enabled 过滤**：YAML 顶层若有 `webui.enabled: false`，则**不显示**（默认 `true`）；
4. **Malformed YAML**：语法错误的不显示，stderr 打 warning。

当前 `src_next/profiles/` 下扫描后**只显示 2 个**：

```text
yellow_qwen3http_cosyvoicehttp    （Gemma4 + Qwen3 VoiceDesign + CosyVoice3）
yellow_qwen3http_indexttshttp     （Gemma4 + Qwen3 VoiceDesign + IndexTTS）
```

### 显示名优先级（从高到低）

```text
1. webui.display_name      （yaml 内 webui 子块）
2. display_name            （yaml 顶层）
3. name                    （yellow_* profile 用此字段）
4. filename stem           （最终 fallback）
```

### Description 优先级（从高到低）

```text
1. webui.description       （yaml 内 webui 子块）
2. description             （yaml 顶层；多行 literal 取首段并截断到 200 字符）
3. 自动摘要                （[region] filename_stem）
```

> **重要**：当前 `src_next/profiles/` 下的 yaml 都**没有** `webui` 子块，也没顶层 `display_name`；下拉框显示的友好名实际取自 `name` 字段（如 `yellow_qwen3http_cosyvoicehttp`）。后续如需更友好的中文名，可在 yaml 中加 `webui.display_name` 字段，**不需要改代码**。

---

## 8. 生成过程说明

点击「🎬 生成有声书」后，页面右侧「生成进度」面板会实时显示 **10 个阶段**的状态（✅ 完成 / ⏳ 进行中 / ⬜ 等待 / ❌ 失败 / 复用）。

### 实际阶段名（与代码一致）

| 序号 | stage name（代码内） | 显示标签 |
|---|---|---|
| 1 | `build_segments` | 文本切分 |
| 2 | `create_llm_client` | LLM 客户端 |
| 3 | `quote_classifier` | 引号分类 |
| 4 | `story_resolver` | 说话人识别 |
| 5 | `character_analyzer` | 角色分析 |
| 6 | `voicebank` | 音色生成 |
| 7 | `story_director` | 导演分析 |
| 8 | `tts_instruction_builder` | TTS 指令 |
| 9 | `tts_synthesis` | TTS 合成 |
| 10 | `audio_merger` | 音频合并 |

> 注：代码实际有 10 个阶段（含 `create_llm_client`），不是 9 个。`create_llm_client` 是初始化 LLM 客户端，通常 < 1 秒；TTS 合成是最耗时的（视段数而定，可能数十秒到数分钟）。

### 实时日志

页面底部「实时日志」框会随阶段进展滚动更新：

- `[stage_name] done in N.NNs, extra_info` — 单个阶段完成；
- `[stage_name] reused from xxx.json, N.NNs` — 复用已有中间结果；
- 最终结果出现后，日志框替换为完整 `pipeline.log` 内容（截断到最后 5000 字符）。

---

## 9. 生成结果说明

### 成功后页面展示

- **音频播放器** — 可直接在网页中试听；
- **下载音频按钮** — 下载最终 wav 文件；
- **结果信息 bar** — 灰色信息条，包含：
  - 生成状态（✅ 生成完成）
  - 总耗时（如 `3m 38s`）
  - 音频时长（如 `2m 47s`）
  - RTF（如 `1.33x`）
  - **Profile** 名称
  - **Task ID**（如 `20260625_143022`）
  - **任务目录**（相对路径，如 `小红帽/20260625_143022`）
  - **音频文件**名（如 `小红帽.wav`）
- **下载音频按钮** — 整块绿色 bar，点击直接下载 wav 文件。

### RTF 是什么

```text
RTF = 总生成耗时 / 最终音频时长
```

例：生成耗时 223s，最终音频时长 167s → RTF = 1.33x。

**RTF 越小，说明生成速度越快**。RTF > 1 表示生成比音频播放慢；RTF < 1 表示比实时快（多段并发合成时可能达到）。

### 失败时

- 顶部红色错误条显示错误类型 + 截断的 traceback（最后 1500 字符）；
- 进度面板中失败的阶段显示 ❌；
- 任务目录仍会创建，`logs/pipeline.log` + `pipeline_result.json` 都会写入失败前的状态；
- 生成按钮恢复可用。

---

## 10. 输出目录结构

### WebUI 任务目录布局

WebUI **不使用** profile 里的 `output.root`，而是固定走自己的根目录，避免和 CLI 跑的产物混在一起：

```text
output-src-next-webui/                  ← 项目根下与 output-src-next/ 同级
└── <profile_stem>/                     ← yaml 文件名（无 .yaml 扩展名）
    └── <task_id>/                      ← 时间戳，如 20260625_143022
        ├── input/
        │   └── <story_name>.txt        ← WebUI 把 textbox 内容落盘到这里
        ├── json/
        │   ├── segments_raw.json             ← stage 1 输出
        │   ├── segments_after_quote_merge.json ← stage 3 输出
        │   ├── quote_classifications.json    ← stage 3 调试产物
        │   ├── resolved_segments.json        ← stage 4 输出
        │   ├── characters.json               ← stage 5 输出
        │   ├── voicebank_result.json         ← stage 6 输出
        │   ├── director_plan.json            ← stage 7 输出
        │   ├── tts_instructions.json         ← stage 8 输出
        │   ├── audio_segment_results.json    ← stage 9 输出（含失败段信息）
        │   ├── audio_result.json             ← stage 10 输出
        │   └── pipeline_result.json          ← 最终结构化结果
        ├── audio_segments/                   ← 分段 wav（实际目录名取自 profile.tts.output_subdir）
        │   └── seg_001.wav ...
        ├── audio_final/
        │   └── <story_name>.wav              ← 最终合成 wav
        ├── voicebank/                        ← 角色音色 wav（实际目录名取自 profile.voicebank.output_subdir）
        │   └── narrator.wav / 小红帽.wav ...
        └── logs/
            └── pipeline.log                  ← 完整日志（含 ISO 时间戳）
```

### 路径示例

假设 profile 是 `src_next/profiles/yellow_qwen3http_cosyvoicehttp.yaml`，task_id 是 `20260625_143022`，story_name 是「小红帽」：

```text
output-src-next-webui/yellow_qwen3http_cosyvoicehttp/20260625_143022/
├── input/小红帽.txt
├── json/pipeline_result.json
├── audio_final/小红帽.wav
└── logs/pipeline.log
```

### 路径各层来源

| 层 | 值 | 来源 |
|---|---|---|
| 1 | `output-src-next-webui/` | WebUI 代码常量 `WEBUI_OUTPUT_ROOT`，**固定** |
| 2 | `<profile_stem>/` | yaml 文件名 stem，如 `yellow_qwen3http_cosyvoicehttp` |
| 3 | `<task_id>/` | `now_timestamp()`，形如 `YYYYMMDD_HHMMSS`；同秒冲突追加 3 位毫秒戳（如 `20260625_143022_456`） |
| 4+ | `input/` / `json/` / `audio_segments/` / `audio_final/` / `voicebank/` / `logs/` | pipeline 内部固定 |

### story_name 去哪了？

**story_name 不进路径**，仅作文件名：

- `input/<story_name>.txt` — WebUI 落盘的输入文件名；
- `audio_final/<story_name>.wav` — 最终音频文件名。

`<story_name>` 取自：

1. 上传 TXT 时 → 文件名 stem；
2. 文本框输入时 → 文本首行前 30 字符（经 `safe_story_name` 清理，仅保留中文 / 字母 / 数字 / `_` / `-`）；
3. 兜底为 `webui_story`。

### 与 CLI 输出目录的区别

| 场景 | 路径 | 说明 |
|---|---|---|
| **WebUI** | `output-src-next-webui/<profile_stem>/<task_id>/` | 强制走自己的根 + task_id 时间戳隔离 |
| **CLI** | `<output.root>/<story_name>/` | 用 profile 的 `output.root`；无 task_id 层；按 story_name 组织 |

两者**完全分离**：CLI 跑的产物在 `output-src-next/`（或 profile 自定义 root）下，WebUI 跑的产物在 `output-src-next-webui/` 下，互不干扰。

---

## 11. 日志路径说明

### 两类日志

| 日志 | 路径 | 内容 | 来源 |
|---|---|---|---|
| 服务进程日志 | `webui.log`（启动时重定向） | 服务启动、Gradio 报错、进程级异常 | `nohup ... > webui.log 2>&1 &` |
| 单次任务日志 | `<task_dir>/logs/pipeline.log` | 10 个阶段的开始 / 完成 / 耗时 / 失败原因 / Summary | `StageLogger` 自动写入 |

### 查看服务进程日志

```bash
tail -f webui.log
```

### 查看单次任务日志

```bash
tail -f <task_dir>/logs/pipeline.log
```

例：

```bash
tail -f output-src-next-webui/yellow_qwen3http_cosyvoicehttp/20260625_143022/logs/pipeline.log
```

### pipeline.log 格式

每行带 ISO 时间戳前缀，例：

```text
2026-06-25T14:30:22 [Pipeline] input=/.../input/小红帽.txt
2026-06-25T14:30:22 [Pipeline] profile=src_next/profiles/yellow_qwen3http_cosyvoicehttp.yaml
2026-06-25T14:30:22 [Pipeline] output=/.../小红帽/20260625_143022
2026-06-25T14:30:22
2026-06-25T14:30:22 [1/10] build_segments ...
2026-06-25T14:30:22 [1/10] build_segments ... done in 0.02s, segments=21
...
2026-06-25T14:35:05 [Summary]
2026-06-25T14:35:05 success=true
2026-06-25T14:35:05 total_time=223.10s
...
```

---

## 12. pipeline_result.json 说明

### 作用

`<task_dir>/json/pipeline_result.json` 是单次任务的**结构化结果摘要**，适合排查、复盘、自动化读取。无论成功还是失败都会写入（失败时含失败前的 stage 信息）。

### 字段（实际代码定义）

```text
{
  "story_name": str,
  "output_dir": str,
  "final_audio": str | None,        # 最终音频绝对路径；失败时可能为 null
  "success": bool,
  "stage_timings": {stage_name: seconds},
  "artifacts": {logical_name: file_path},  # 11 个中间产物的路径
  "error": str,                     # 空字符串表示成功
  "pipeline_summary": {
    "total_time_sec": float,
    "analysis_time_sec": float,     # quote_classifier + story_resolver + character_analyzer + story_director
    "voicebank_time_sec": float,
    "tts_time_sec": float,
    "merge_time_sec": float,
    "final_audio_duration_sec": float | None,
    "rtf": float | None,
    "output_dir": str,
    "final_audio_path": str,
    "stages": [
      {
        "stage": str,
        "status": "success" | "failed",
        "elapsed_sec": float,
        "mode": "run" | "reused" | "cached",
        "output": str,              # 该 stage 输出文件路径
        "error": str | None         # 失败 stage 才有
      },
      ...
    ]
  }
}
```

### 与本文档第一条命令清单的差异说明

- 顶层**没有** `profile` 字段（profile 信息不在 `PipelineResult` 数据模型里，可从 `output_dir` 路径反推）；
- `final_audio` 字段名是 `final_audio`，**不是** `final_audio_path`（后者是 `pipeline_summary` 内部的字段名）；
- 时间字段（`*_time_sec`）都在 `pipeline_summary` 子对象里，**不**在顶层；
- `stages` 数组每项含 `mode` 字段（run / reused / cached），用于区分本次执行还是命中缓存。

### artifacts 字段（中间产物路径）

```text
{
  "segments_raw": ".../json/segments_raw.json",
  "segments_after_quote_merge": ".../json/segments_after_quote_merge.json",
  "quote_classifications": ".../json/quote_classifications.json",   # 仅 stage 3 实际生成时
  "resolved_segments": ".../json/resolved_segments.json",
  "characters": ".../json/characters.json",
  "voicebank_result": ".../json/voicebank_result.json",
  "director_plan": ".../json/director_plan.json",
  "tts_instructions": ".../json/tts_instructions.json",
  "audio_segment_results": ".../json/audio_segment_results.json",
  "audio_result": ".../json/audio_result.json",
  "pipeline_result": ".../json/pipeline_result.json"
}
```

---

## 13. 常见问题排查

### 13.1 黄区打不开网页

依次检查：

```bash
ps aux | grep gradio_webui       # 进程是否在
tail -f webui.log                # 启动是否报错
lsof -i:7860                     # 端口是否监听（或 netstat -tulnp | grep 7860）
curl -I http://127.0.0.1:7860    # 本机是否能 200
```

可能原因：

- WebUI 进程未启动 / 已崩溃；
- 监听端口不是 7860（启动参数 `--port` 改过）；
- 启动时 Python 报错（缺依赖 / 缺 yaml / 缺 `src_next` 模块）；
- 服务器防火墙未放行 7860；
- 黄区浏览器访问地址写错（应是 `http://10.50.121.102:7860`，注意是 `http` 不是 `https`）。

### 13.2 profile 下拉框为空 / 选项很少

检查：

```bash
ls src_next/profiles/*.yaml | head -20
```

可能原因：

- `src_next/profiles/` 下没有**完整** profile（缺 `llm` / `voicebank` / `tts` / `output` / `pipeline` 任一块的会被过滤）；
- 某个 yaml 语法错误（YAML parse 失败，stderr 会打 warning）；
- 某个 yaml 设了 `webui.enabled: false`；
- WebUI 启动时 `src_next/profiles/` 路径解析错误（检查 cwd）。

调试命令（直接看扫描结果）：

```bash
python -c "from src_next.utils.yaml_utils import discover_profiles; import json; print(json.dumps(discover_profiles('src_next/profiles'), ensure_ascii=False, indent=2))"
```

### 13.3 点击「生成」后很久没反应

可能原因：

- 正在**排队**（最多 5 个生成任务同时运行，超过会进队列）；
- 某个阶段在调用 LLM / TTS 服务，耗时较长（特别是 stage 9 `tts_synthesis` 视段数可能跑几十秒到几分钟）；
- LLM / TTS 服务器无响应（ hung 死）。

排查：

```bash
tail -f webui.log                                      # 服务进程日志
tail -f <task_dir>/logs/pipeline.log                   # 当前任务日志（看卡在哪个 stage）
```

### 13.4 生成失败

按以下顺序排查：

1. **页面顶部红色错误条** — 直接看错误类型 + traceback 末尾；
2. **实时日志** — 看哪个 stage 失败；
3. **任务目录下的完整日志**：

   ```bash
   cat <task_dir>/logs/pipeline.log | tail -100
   ```

4. **结构化结果**：

   ```bash
   cat <task_dir>/json/pipeline_result.json | python -m json.tool
   ```

   看 `error` 字段和 `pipeline_summary.stages` 数组里 `status=="failed"` 的项；

5. **TTS 分段结果**（如果失败发生在 stage 9 或之后）：

   ```bash
   cat <task_dir>/json/audio_segment_results.json | python -m json.tool | head -50
   ```

   每段含 `success` 字段和失败原因。

### 13.5 音频播放器没出现

可能原因：

- `tts_synthesis` 全部段失败；
- `audio_merger` 失败（无成功段可拼）；
- 最终音频路径权限问题（写了但读不出）；
- `final_audio` 路径不存在（极少数情况，检查 `pipeline_result.json`）。

排查：

```bash
ls -lh <task_dir>/audio_final/
ls -lh <task_dir>/audio_segments/ | head -5
cat <task_dir>/json/pipeline_result.json | python -c "import json, sys; r = json.load(sys.stdin); print('final_audio:', r.get('final_audio'))"
```

---

## 14. 多用户使用说明

### 并发限制

- 默认**最多 5 个生成任务同时运行**（`--concurrency 5`）；
- 超过并发数的任务会自动排队，队列上限 20（`--queue-size 20`）；
- 队列满时新点击「生成」会被告知排队失败（Gradio 默认行为）；
- **打开网页本身不占并发**，只有点击「生成」才占用一个生成槽位。

### 任务目录隔离

- 每次点击「生成」都会创建**独立 task_id**（基于当前时间戳，如 `20260625_143022`）；
- 任务目录互不覆盖：`output-src-next-webui/<profile_stem>/<task_id>/`；
- 即使两个用户用相同的 story_name，也不会互相覆盖（不同 task_id 子目录）；
- **与 CLI 跑的 `output-src-next/<story_name>/` 完全分离**，互不干扰。

---

## 15. 开发者结构说明

新 WebUI 涉及的层（按调用链自上而下）：

| 文件 | 职责 |
|---|---|
| `src_next/app/gradio_webui.py` | Gradio 页面、交互、profile 下拉框、输入校验、事件 yield、结果展示。**WebUI 入口** |
| `src_next/core/audiobook_pipeline.py` | 新链路 pipeline 编排。两个入口：`run_pipeline`（CLI 同步） + `run_pipeline_stream`（WebUI 生成器版）；共享 `_prepare_paths` + `_build_pipeline_result` + `_load_pipeline_profile` 等 helper |
| `src_next/core/logging_utils.py` | `StageLogger` 类（终端 print + 文件落盘 + 内存累积三合一）；旧 `log_stage_*` 函数保留给 mock pipeline |
| `src_next/utils/file_utils.py` | 文本读取（编码 fallback）、文件大小检查、`safe_story_name`、`save_text` / `save_json_file` |
| `src_next/utils/yaml_utils.py` | `load_yaml` + `discover_profiles`（含 webui 块过滤 + 优先级 resolution） |
| `src_next/utils/time_utils.py` | `format_seconds` + `now_timestamp`（task_id 来源） |

### WebUI 调用链

```text
用户点击「生成有声书」
    ↓
generate_audiobook_handler()              ← gradio_webui.py
    ↓ 校验输入 → 写 input/<story>.txt → 禁用按钮
    ↓
for event in run_pipeline_stream(...):    ← audiobook_pipeline.py
    ↓
    yield gr.update(...) × 13              ← 实时刷新页面
    ↓
最终 {"type": "result"} 或 {"type": "error"} 事件
    ↓
显示 audio_player / download_btn / result_info
```

### Event 流（run_pipeline_stream yields）

| 事件类型 | 触发时机 | 携带字段 |
|---|---|---|
| `stage_start` | 每个 stage 开始 | `step`, `name`, `stage_index`, `total_stages`, `timestamp` |
| `stage_done` | stage 正常完成 | `step`, `name`, `stage_index`, `elapsed_sec`, `extra`, `cumulative_sec` |
| `stage_reused` | reuse_existing 命中缓存 | `step`, `name`, `stage_index`, `src`, `elapsed_sec` |
| `stage_failed` | 单 stage 失败但未阻断 | `step`, `name`, `stage_index`, `elapsed_sec`, `error` |
| `result` | 全部完成 | `result: PipelineResult`, `logs: str` |
| `error` | 不可恢复异常 | `error: str`, `traceback: str`, `result: PipelineResult`, `logs: str` |

---

## 16. 常用命令速查

```bash
# ─── 启动 ────────────────────────────────────────────────────────
# 前台启动（调试）
python -m src_next.app.gradio_webui --host 0.0.0.0 --port 7860 --concurrency 5 --queue-size 20

# 后台启动（服务器常驻）
nohup python -m src_next.app.gradio_webui \
    --host 0.0.0.0 --port 7860 \
    --concurrency 5 --queue-size 20 \
    > webui.log 2>&1 &

# ─── 查看 ────────────────────────────────────────────────────────
# 服务进程日志
tail -f webui.log

# 进程状态
ps aux | grep gradio_webui

# 端口监听
lsof -i:7860
# 或
netstat -tulnp | grep 7860

# 本机探活
curl -I http://127.0.0.1:7860

# ─── 单次任务日志 ────────────────────────────────────────────────
tail -f <task_dir>/logs/pipeline.log

# 查看 pipeline_result.json（结构化结果）
cat <task_dir>/json/pipeline_result.json | python -m json.tool

# ─── Profile 调试 ────────────────────────────────────────────────
# 看 WebUI 扫描出了哪些 profile
python -c "from src_next.utils.yaml_utils import discover_profiles; import json; print(json.dumps(discover_profiles('src_next/profiles'), ensure_ascii=False, indent=2))"

# ─── 关闭 ────────────────────────────────────────────────────────
kill <PID>
kill -9 <PID>           # 强制

# ─── 访问 ────────────────────────────────────────────────────────
# 黄区浏览器
http://10.50.121.102:7860
```

---

## 维护说明

本文档与以下代码强耦合，**代码变更时需同步更新本文档**：

| 文档章节 | 关联代码 | 需要同步的场景 |
|---|---|---|
| 第 6 节输入限制 | `gradio_webui.py` `MAX_TEXT_LENGTH` / `MAX_TXT_FILE_SIZE_BYTES` 常量 | 调整限制值 |
| 第 7 节 profile 发现规则 | `utils/yaml_utils.py` `discover_profiles` + `_REQUIRED_BLOCKS` | 增删 required 块 / 改优先级 |
| 第 8 节阶段列表 | `gradio_webui.py` `STAGE_DISPLAY` + `audiobook_pipeline.py` 各 stage | 增删 stage |
| 第 10 节目录布局 | `audiobook_pipeline.py` `_prepare_paths` + 各 stage `_save_json` 路径 | 改路径 |
| 第 12 节字段 | `data_models.py` `PipelineResult` + `audiobook_pipeline.py` `_build_pipeline_summary` | 加减字段 |
| 第 14 节并发参数 | `gradio_webui.py` `_parse_args` 默认值 | 改默认 |
| 第 15 节调用链 | `run_pipeline_stream` + event yield 位置 | 改事件 schema |
