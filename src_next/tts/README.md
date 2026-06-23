# tts/ TTS 合成适配层

> 整体架构请见 [../README.md](../README.md) 的「〇、一图看懂 src_next」。本层在数据流中的位置：
>
> ```
> analysis层 → core/tts_instruction_builder → 【tts层】→ core/audio_merger
>     TTSInstruction + VoicebankResult              list[AudioSegmentResult]
> ```

## 一、这一层负责什么

* 把通用 `TTSInstruction` 翻译成具体 TTS 后端（IndexTTS / CosyVoice / FishPro / Qwen TTS / mock）的调用。
* 调用后端合成每段 wav，按 segment 顺序返回 `list[AudioSegmentResult]`。
* 单条失败隔离：失败 instruction 写到对应 `AudioSegmentResult.error`，不阻断其他条。
* 缓存：已存在的 wav 可复用，dry_run 模式只写 invocation 不调用模型。

## 二、这一层不负责什么

* 不切分文本（`core/segment_builder`）。
* 不识别说话人（`analysis/story_resolver`）。
* 不抽取角色（`analysis/character_analyzer`）。
* 不生成导演指令（`analysis/story_director`）。
* 不准备音色参考（`voicebank/`）。
* 不拼接多段（`core/audio_merger`）。
* 不解析 yaml profile（`profiles/`）。

## 三、统一接口

```python
class BaseTTSAdapter(ABC):
    def synthesize(
        self,
        instructions: list[TTSInstruction],
        voicebank_result: VoicebankResult,
        output_dir: str,
        **kwargs: Any,
    ) -> list[AudioSegmentResult]: ...
```

约定：

* 输入长度 = 输出长度，按 instruction 顺序一一对应；
* `voicebank_result` 既用于查 speaker → voice_ref（fallback），也用于审计；
* `output_dir` 是故事级输出根目录；adapter 在其下创建 `output_subdir` 子目录放 wav + log；
* `**kwargs` 至少接受 `dry_run: bool` 和 `limit: int`（0 = 全量，>0 = 仅前 N 条）。

## 四、当前 backend 实现状态

| backend | 文件 | 状态 |
|---|---|---|
| mock | `mock_tts.py` | ✅ 可用，离线占位 |
| indextts | `indextts_adapter.py` | ✅ 可用，subprocess 调外部 IndexTTS CLI |
| cosyvoice | `cosyvoice_adapter.py` | 🚧 占位（`NotImplementedError`） |
| fishpro | `fishpro_adapter.py` | 🚧 占位（`NotImplementedError`） |
| qwen_tts | `qwen_tts_adapter.py` | 🚧 占位（`NotImplementedError`） |

## 五、IndexTTS adapter 详解

### 5.1 真实接口（参考 `indextts/cli.py`）

```bash
python -m indextts.cli <text>
    -v, --voice <wav path>           (required)
    -o, --output_path <wav path>     (default "gen.wav")
    -c, --config <yaml path>         (default "checkpoints/config.yaml")
    --model_dir <dir>                (default "checkpoints")
    --fp16                           (default True)
    -f, --force                      overwrite
    -d, --device <cpu|cuda:0|mps>    (default auto)
```

**IndexTTS 是纯 zero-shot voice cloning**：
* 文本 + 参考音频 → 输出 wav，仅此而已；
* 不支持 pace / emotion / volume / pitch / stress_words / delivery_instruction；
* 这些通用字段在 `TTSInstruction` 里保留是为了让其它后端（如 CosyVoice2）能用，
  IndexTTS adapter 会把它们写到 per-segment log 留档，**不影响合成结果**。

### 5.2 构造参数

```python
IndexTTSAdapter(
    engine_root: str,                  # indextts 仓库根目录
    script_path: str | None = None,    # 不传 → 默认 "-m indextts.cli"
    model_path: str | None = None,     # 不传 → 默认 "checkpoints/"
    config_path: str | None = None,    # 不传 → 默认 "checkpoints/config.yaml"
    python_executable: str | list[str] | None = None,  # 支持 WSL 包装
    output_subdir: str = "audio_segments",
    extra_args: dict[str, Any] | None = None,  # device / disable_fp16 / timeout_per_seg
)
```

### 5.3 WSL 路径转换

和 `voicebank/qwen_voicegenerator.py` 完全一致：`python_executable` 首段是 `wsl` 时，
adapter 自动把绝对路径 `F:\\...` → `/mnt/f/...` 再传给 subprocess。

### 5.4 单条调用流程

```
for inst in instructions:
    voice_ref = inst.voice_ref or speaker_to_voice[inst.speaker]
    if missing voice_ref: → success=False, 不调用
    if output wav exists & non-empty: → 复用，success=True
    if dry_run: → 只写 log，success=False
    else: subprocess(cli) → check wav → success=True / False
```

失败列表汇总写到 `<audio_dir>/errors.log`；adapter 配置快照写到
`<audio_dir>/adapter_config.json`。

## 六、如何测试

### 6.1 不重跑 analysis 的 smoke test

`test_tts_from_artifacts.py` 直接读已经落盘的 artifacts：

```bash
# mock 验证数据流
python -m src_next.tts.test_tts_from_artifacts \
    --artifact-dir output-src-next-analysis-test/桂花雨 \
    --backend mock

# IndexTTS dry-run（不调模型）
python -m src_next.tts.test_tts_from_artifacts \
    --artifact-dir output-src-next-analysis-test/桂花雨 \
    --backend indextts \
    --dry-run true \
    --limit 2

# IndexTTS 真实合成（先少量）
python -m src_next.tts.test_tts_from_artifacts \
    --artifact-dir output-src-next-analysis-test/桂花雨 \
    --backend indextts \
    --dry-run false \
    --limit 2

# 全量
python -m src_next.tts.test_tts_from_artifacts \
    --artifact-dir output-src-next-analysis-test/桂花雨 \
    --backend indextts \
    --dry-run false \
    --limit 0
```

### 6.2 输出结构

```
<artifact_dir>/
├── json/
│   └── audio_segment_results.json   ← 新增
└── audio_segments/                  ← 新增
    ├── adapter_config.json
    ├── errors.log                   ← 仅失败时存在
    ├── seg_001.wav
    ├── seg_001.log
    ├── seg_002.wav
    ├── seg_002.log
    └── ...
```

每条 `.log` 文件包含：
* `=== INVOCATION ===`：完整命令行 + cwd；
* `=== STYLE (NOT passed to IndexTTS; logged only) ===`：通用风格字段留档；
* `=== OUTPUT ===`：IndexTTS CLI 的 stdout/stderr。

### 6.3 py_compile 自检

```bash
python -m py_compile \
    src_next/tts/base.py \
    src_next/tts/mock_tts.py \
    src_next/tts/indextts_adapter.py \
    src_next/tts/cosyvoice_adapter.py \
    src_next/tts/fishpro_adapter.py \
    src_next/tts/qwen_tts_adapter.py \
    src_next/tts/registry.py \
    src_next/tts/test_tts_from_artifacts.py
```

## 七、后续 TODO

* IndexTTS 每条都重新加载模型，慢。可以写一个 `src_next/tts/scripts/run_indextts_batch.py`
  wrapper：启动时加载一次模型，从 stdin / JSON 读 batch，逐条合成。这样可以
  从 17 段 × ~30s → 17 段 × ~5s。
* CosyVoice2 支持 `instruct_text`，把 emotion / volume / pace 以自然语言 prompt
  传过去（比 IndexTTS 表达力强很多）。
* IndexTTS 支持标点控制停顿——可以在 adapter 里按 `pause_hint` 自动插入
  逗号 / 句号，近似模拟 pace。
* `core/audio_merger.py` 已支持 stdlib `wave` 直接拼接，但没做重采样；
  如果不同段采样率不同会跳过——TODO 接 torchaudio 重采样。
