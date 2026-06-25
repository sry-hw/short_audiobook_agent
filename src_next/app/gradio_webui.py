"""src_next/app/gradio_webui.py

src_next 有声书生成 WebUI。

底层调用 ``src_next.core.audiobook_pipeline.run_pipeline_stream``（生成器版），
按 stage 事件实时刷新页面。

启动方式（开发）::

    python -m src_next.app.gradio_webui --host 0.0.0.0 --port 7860

启动方式（服务器常驻）::

    nohup python -m src_next.app.gradio_webui \\
        --host 0.0.0.0 --port 7860 \\
        --concurrency 5 --queue-size 20 \\
        > webui.log 2>&1 &

黄区访问：``http://10.50.121.102:7860``
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import traceback as tb_mod
from pathlib import Path
from typing import Any

# Windows GBK 终端兼容：把 stdout/stderr 切成 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr  # noqa: E402

# 业务 import 放在 wrapper 之后
from src_next.core.audiobook_pipeline import run_pipeline_stream  # noqa: E402
from src_next.core.data_models import PipelineResult  # noqa: E402
from src_next.utils.file_utils import (  # noqa: E402
    file_size_ok,
    read_text_with_encoding,
    safe_story_name,
)
from src_next.utils.time_utils import format_seconds, now_timestamp  # noqa: E402
from src_next.utils.yaml_utils import discover_profiles, load_yaml  # noqa: E402


# ─── 常量 ────────────────────────────────────────────────────────────────────

MAX_TEXT_LENGTH = 3500
MAX_TXT_FILE_SIZE_BYTES = 20 * 1024  # 20 KB
DEFAULT_CONCURRENCY = 5
DEFAULT_QUEUE_SIZE = 20

# WebUI 任务输出根目录（与 CLI 的 output-src-next/ 同级，互不干扰）。
# 实际任务路径：``output-src-next-webui/<profile_stem>/<task_id>/``。
# story_name 不进路径，仅作文件名（input/<story>.txt、audio_final/<story>.wav）。
WEBUI_OUTPUT_ROOT = "output-src-next-webui"

_PROFILES_DIR = "src_next/profiles"

# 10 stages（与 audiobook_pipeline.run_pipeline_stream 一致）
STAGE_DISPLAY: list[tuple[str, str]] = [
    ("build_segments",          "文本切分"),
    ("create_llm_client",       "LLM 客户端"),
    ("quote_classifier",        "引号分类"),
    ("story_resolver",          "说话人识别"),
    ("character_analyzer",      "角色分析"),
    ("voicebank",               "音色生成"),
    ("story_director",          "导演分析"),
    ("tts_instruction_builder", "TTS 指令"),
    ("tts_synthesis",           "TTS 合成"),
    ("audio_merger",            "音频合并"),
]

ICON_DONE = "✅"
ICON_RUNNING = "⏳"
ICON_WAITING = "⬜"
ICON_FAILED = "❌"

# CSS 样式
CUSTOM_CSS = """
/* 紧凑排版：减小默认块间距 */
.gr-block {
    margin-bottom: 6px !important;
}
.gr-row {
    gap: 8px !important;
}
.gradio-container .gr-block > .wrap {
    padding: 6px !important;
}

/* 结果信息 bar */
.result-bar {
    background: #f0f7ff;
    border: 1px solid #bae0ff;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    margin-top: 4px;
    line-height: 1.5;
}

/* 阶段进度面板 */
.stage-panel {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: monospace;
    font-size: 12.5px;
    line-height: 1.5;
}

/* 错误提示 */
.error-text {
    background: #fff5f5;
    border: 1px solid #ffccc7;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cf1322;
    font-size: 13px;
    line-height: 1.5;
}

/* 下载音频按钮：整块 bar 样式 */
.download-bar {
    width: 100% !important;
    background: linear-gradient(135deg, #52c41a 0%, #389e0d 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    text-align: center !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(82, 196, 26, 0.3) !important;
    transition: all 0.2s !important;
    display: block !important;
}
.download-bar:hover {
    background: linear-gradient(135deg, #389e0d 0%, #237804 100%) !important;
    box-shadow: 0 4px 10px rgba(82, 196, 26, 0.4) !important;
    transform: translateY(-1px) !important;
}
"""


# ─── Stage panel 渲染 ───────────────────────────────────────────────────────


def _render_stage_panel_initial() -> str:
    """初始状态：所有 stage 等待。"""
    lines = []
    for i, (_, label) in enumerate(STAGE_DISPLAY):
        lines.append(f"{ICON_WAITING} [{i + 1}/10] {label} 等待")
    return "\n".join(lines)


def _render_stage_panel_from_events(events: list[dict[str, Any]]) -> str:
    """根据收集到的 stage 事件渲染进度面板。"""
    # 每个阶段的最新状态：done / reused / failed / running / waiting
    stage_status: dict[str, str] = {}
    current_running: str | None = None
    failed_stage: str | None = None

    for evt in events:
        etype = evt.get("type")

        # error 事件可能不带 name；优先处理（标记当前运行 stage 为失败）
        if etype == "error":
            if current_running:
                stage_status[current_running] = "failed"
                failed_stage = current_running
                current_running = None
            continue

        name = evt.get("name")
        if not name:
            continue

        if etype == "stage_start":
            stage_status[name] = "running"
            current_running = name
        elif etype == "stage_done":
            stage_status[name] = "done"
            if current_running == name:
                current_running = None
        elif etype == "stage_reused":
            stage_status[name] = "reused"
            if current_running == name:
                current_running = None
        elif etype == "stage_failed":
            stage_status[name] = "failed"
            failed_stage = name
            if current_running == name:
                current_running = None

    lines = []
    for i, (name, label) in enumerate(STAGE_DISPLAY):
        status = stage_status.get(name, "waiting")
        if status == "done":
            icon = ICON_DONE
            tag = "完成"
        elif status == "reused":
            icon = ICON_DONE
            tag = "复用"
        elif status == "running":
            icon = ICON_RUNNING
            tag = "进行中"
        elif status == "failed":
            icon = ICON_FAILED
            tag = "失败"
        else:
            icon = ICON_WAITING
            tag = "等待"
        lines.append(f"{icon} [{i + 1}/10] {label} {tag}")

    return "\n".join(lines)


def _render_stage_panel_done() -> str:
    """全部完成。"""
    lines = []
    for i, (_, label) in enumerate(STAGE_DISPLAY):
        lines.append(f"{ICON_DONE} [{i + 1}/10] {label} 完成")
    return "\n".join(lines)


# ─── 结果栏 + 日志摘要 ──────────────────────────────────────────────────────


def _render_result_bar(
    pipeline_result: PipelineResult,
    profile_display_name: str,
    task_id: str,
) -> str:
    """渲染结果信息 bar。"""
    summary = pipeline_result.pipeline_summary or {}
    total_time = summary.get("total_time_sec") or 0
    fa_dur = summary.get("final_audio_duration_sec")
    rtf = summary.get("rtf")
    audio_name = Path(pipeline_result.final_audio).name if pipeline_result.final_audio else "未生成"

    parts = [
        f"✅ 生成完成 | 总耗时: {format_seconds(total_time)}",
    ]
    if fa_dur is not None:
        parts.append(f"音频时长: {format_seconds(fa_dur)}")
    if rtf is not None:
        parts.append(f"RTF: {rtf:.2f}x")
    bar_line = " | ".join(parts)

    output_dir_name = Path(pipeline_result.output_dir).name
    parent_name = Path(pipeline_result.output_dir).parent.name

    return (
        f"{bar_line}\n\n"
        f"**Profile**: {profile_display_name}  \n"
        f"**Task ID**: `{task_id}`  \n"
        f"**任务目录**: `{parent_name}/{output_dir_name}`  \n"
        f"**音频文件**: `{audio_name}`"
    )


# ─── 输入校验 + story_name 推断 ────────────────────────────────────────────


def _is_task_id(name: str) -> bool:
    """task_id 格式：``YYYYMMDD_HHMMSS`` 或带后缀。"""
    return bool(re.match(r"^\d{8}_\d{6}(_[A-Za-z0-9]+)?$", name))


def _derive_story_name(text: str, file_upload) -> str:
    """从上传文件名或文本首行推断 story_name。"""
    if file_upload is not None and getattr(file_upload, "name", None):
        return safe_story_name(Path(file_upload.name).stem)
    first_line = (text.strip().split("\n")[0])[:30] if text and text.strip() else ""
    return safe_story_name(first_line) if first_line else "webui_story"


# ─── 事件处理器 ──────────────────────────────────────────────────────────────


def on_file_upload(file_obj):
    """处理 TXT 上传：先校验大小，再读编码，再校验字符数。"""
    if file_obj is None:
        return "", gr.update(), gr.update(visible=False)

    file_path = file_obj.name

    # 1. 文件大小校验
    if not file_size_ok(file_path, MAX_TXT_FILE_SIZE_BYTES):
        size_kb = Path(file_path).stat().st_size / 1024
        err = (
            f"❌ 文件过大（{size_kb:.1f} KB），最大 "
            f"{MAX_TXT_FILE_SIZE_BYTES // 1024} KB；请缩短或拆分。"
        )
        return "", gr.update(value=None), gr.update(value=err, visible=True)

    # 2. 编码读取（UTF-8 → GBK fallback）
    content, err = read_text_with_encoding(file_path)
    if err:
        return "", gr.update(value=None), gr.update(
            value=f"❌ 文件读取失败：{err}（请用 UTF-8 或 GBK 保存）",
            visible=True,
        )

    # 3. 字符数校验
    if len(content) > MAX_TEXT_LENGTH:
        return "", gr.update(value=None), gr.update(
            value=(
                f"❌ 文件内容超出 {MAX_TEXT_LENGTH} 字限制"
                f"（当前 {len(content)} 字），请缩短。"
            ),
            visible=True,
        )

    # 4. 成功：填入 textbox，清空错误
    return content, gr.update(), gr.update(visible=False)


def on_text_change(text: str):
    """字数显示。"""
    if not text:
        return gr.update(value=f"0 / {MAX_TEXT_LENGTH}")
    n = len(text)
    if n > MAX_TEXT_LENGTH:
        return gr.update(value=f"⚠️ {n} / {MAX_TEXT_LENGTH}（已超限）")
    return gr.update(value=f"{n} / {MAX_TEXT_LENGTH}")


def on_profile_change(profile_path: str):
    """切换 profile 时更新描述。"""
    if not profile_path:
        return gr.update(value="")
    try:
        profiles = discover_profiles(_PROFILES_DIR)
    except Exception:
        profiles = []
    for p in profiles:
        if p["path"] == profile_path:
            desc = p.get("description", "")
            region = p.get("region", "")
            header = f"**[{region}]** {p['display_name']}"
            return gr.update(value=f"{header}\n\n{desc}" if desc else header)
    return gr.update(value="")


# ─── 主生成 generator ──────────────────────────────────────────────────────


# 输出 tuple 的字段顺序（必须和事件绑定 outputs 顺序一致）
# 11 个组件：
#   1. error_text         (Markdown, 默认隐藏)
#   2. textbox            (故事文本)
#   3. file_upload        (TXT)
#   4. profile_dropdown   (Dropdown)
#   5. profile_description (Markdown)
#   6. progress_panel     (Textbox)
#   7. audio_player       (Audio, 默认隐藏)
#   8. download_btn       (DownloadButton, 默认隐藏)
#   9. result_info        (Markdown, 默认隐藏)
#  10. log_display        (Textbox)
#  11. generate_btn       (Button)


def _build_yield(
    *,
    error_value=None,        # str | None
    error_visible=None,      # bool | None
    textbox_value=None,
    file_upload_value=None,  # gr.update() or None
    profile_value=None,
    profile_desc_value=None,
    progress_value=None,
    audio_value=None,        # str | None
    audio_visible=None,
    download_value=None,
    download_visible=None,
    result_value=None,
    result_visible=None,
    log_display_value=None,
    btn_interactive=None,
):
    """构造 11 元素 tuple；任何字段为 None 时用 gr.update()（不变化）。"""
    def md(value, visible=None):
        if value is None and visible is None:
            return gr.update()
        kwargs = {}
        if value is not None:
            kwargs["value"] = value
        if visible is not None:
            kwargs["visible"] = visible
        return gr.update(**kwargs)

    def tb(value, visible=None):
        return md(value, visible)

    def fld(value):
        if value is None:
            return gr.update()
        return gr.update(value=value)

    def btn(interactive):
        if interactive is None:
            return gr.update()
        return gr.update(interactive=interactive)

    return (
        md(error_value, error_visible),                                              # 1
        tb(textbox_value),                                                           # 2
        gr.update() if file_upload_value is None else file_upload_value,             # 3
        fld(profile_value),                                                          # 4
        md(profile_desc_value),                                                      # 5
        tb(progress_value),                                                          # 6
        md(audio_value, audio_visible),                                              # 7
        md(download_value, download_visible),                                        # 8
        md(result_value, result_visible),                                            # 9
        tb(log_display_value),                                                       # 10
        btn(btn_interactive),                                                        # 11
    )


def generate_audiobook_handler(
    text: str,
    file_upload,
    profile_path: str,
):
    """主生成函数；generator + yield 11 元素 tuple。

    Inputs 3 个 → 返回 tuple 11 个（与 outputs 列表对应）。
    状态（events_collected / current_log_text）通过闭包维护。
    """
    # ── 1. 输入校验 ─────────────────────────────────────────────────
    if not text or not text.strip():
        yield _build_yield(
            error_value="请输入故事文本", error_visible=True,
            progress_value=_render_stage_panel_initial(),
            btn_interactive=True,
        )
        return

    if len(text) > MAX_TEXT_LENGTH:
        yield _build_yield(
            error_value=(
                f"文本超出 {MAX_TEXT_LENGTH} 字限制（当前 {len(text)} 字）"
            ),
            error_visible=True,
            btn_interactive=True,
        )
        return

    if not profile_path:
        yield _build_yield(
            error_value="请先选择一个 profile",
            error_visible=True,
            btn_interactive=True,
        )
        return

    # ── 2. 加载 profile（用于 LLM/voicebank/tts 配置；output.root 被忽略） ──
    try:
        profile_dict = load_yaml(profile_path)
    except Exception as err:
        yield _build_yield(
            error_value=f"profile 加载失败: {type(err).__name__}: {err}",
            error_visible=True,
            btn_interactive=True,
        )
        return

    # WebUI 强制用自己的 output root，不走 profile['output']['root']
    # （避免和 CLI 跑的产物混在 output-src-next/<story>/ 里）。
    # 路径布局：<WEBUI_OUTPUT_ROOT>/<profile_stem>/<task_id>/
    profile_stem = Path(profile_path).stem
    webui_output_root = str(Path(WEBUI_OUTPUT_ROOT) / profile_stem)

    # ── 3. 推断 story_name + 生成 task_id ───────────────────────────
    story_name = _derive_story_name(text, file_upload)
    task_id = now_timestamp()

    # ── 4. 构造 output_dir + 落盘 input 文件 ────────────────────────
    # task_id_layout=True → output_dir = <webui_output_root>/<task_id>/（无 story_name 层）
    output_dir = Path(webui_output_root).expanduser().resolve() / task_id
    # task_id 同秒冲突保护：dir 已存在则追加 3 位毫秒戳
    if output_dir.exists():
        from datetime import datetime as _dt
        ms_suffix = _dt.now().strftime("%f")[:3]
        task_id = f"{task_id}_{ms_suffix}"
        output_dir = Path(webui_output_root).expanduser().resolve() / task_id
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / f"{story_name}.txt"
    input_file.write_text(text, encoding="utf-8")

    # ── 5. 找 profile 友好名（给 result bar 用） ────────────────────
    profile_display_name = profile_stem
    try:
        for p in discover_profiles(_PROFILES_DIR):
            if p["path"] == profile_path:
                profile_display_name = p["display_name"]
                break
    except Exception:
        pass

    # ── 6. 禁用按钮 + 清空错误 + 初始进度 ──────────────────────────
    yield _build_yield(
        error_visible=False,
        progress_value=_render_stage_panel_initial(),
        log_display_value=(
            f"🚀 开始生成...\n"
            f"Profile: {profile_display_name}\n"
            f"Task ID: {task_id}\n"
            f"Story: {story_name}\n"
            f"输出目录: {output_dir}\n"
        ),
        btn_interactive=False,
    )

    # ── 7. 迭代 stream events ───────────────────────────────────────
    events_collected: list[dict[str, Any]] = []
    current_log_text = ""

    try:
        for event in run_pipeline_stream(
            str(input_file),
            profile_dict,
            output_root=webui_output_root,
            story_name=story_name,
            task_id=task_id,
            task_id_layout=True,
        ):
            events_collected.append(event)
            etype = event.get("type")

            if etype == "stage_start":
                yield _build_yield(
                    progress_value=_render_stage_panel_from_events(events_collected),
                    btn_interactive=False,
                )

            elif etype in ("stage_done", "stage_reused", "stage_failed"):
                # 更新进度面板；日志等到 result/error 才完整刷
                extra = event.get("extra", "")
                src = event.get("src", "")
                stage_name = event.get("name", "")
                elapsed = event.get("elapsed_sec", 0)
                if etype == "stage_done":
                    line = f"[{stage_name}] done in {elapsed:.2f}s, {extra}"
                elif etype == "stage_reused":
                    line = f"[{stage_name}] reused from {src}, {elapsed:.2f}s"
                else:
                    line = f"[{stage_name}] FAILED in {elapsed:.2f}s — {event.get('error', '')}"
                current_log_text += line + "\n"
                yield _build_yield(
                    progress_value=_render_stage_panel_from_events(events_collected),
                    log_display_value=current_log_text,
                    btn_interactive=False,
                )

            elif etype == "result":
                pipeline_result: PipelineResult = event["result"]
                logs = event.get("logs", "") or current_log_text
                # 截断超长日志避免 Gradio textbox 内存爆
                if len(logs) > 5000:
                    logs = logs[-5000:]
                if pipeline_result.final_audio and Path(pipeline_result.final_audio).exists():
                    audio_value = pipeline_result.final_audio
                    audio_visible = True
                    download_value = pipeline_result.final_audio
                    download_visible = True
                else:
                    audio_value = None
                    audio_visible = False
                    download_value = None
                    download_visible = False
                yield _build_yield(
                    progress_value=_render_stage_panel_done(),
                    audio_value=audio_value,
                    audio_visible=audio_visible,
                    download_value=download_value,
                    download_visible=download_visible,
                    result_value=_render_result_bar(
                        pipeline_result, profile_display_name, task_id,
                    ),
                    result_visible=True,
                    log_display_value=logs,
                    btn_interactive=True,
                )
                return

            elif etype == "error":
                pipeline_result = event.get("result")
                error_msg = event.get("error", "未知错误")
                tb_str = event.get("traceback", "")
                logs = event.get("logs", "") or current_log_text
                if len(logs) > 5000:
                    logs = logs[-5000:]
                # 截断 traceback 避免 UI 爆
                tb_short = tb_str[-1500:] if tb_str else ""
                # 如果失败前有部分结果，result bar 也显示
                result_visible = False
                result_value = None
                if pipeline_result is not None:
                    result_value = _render_result_bar(
                        pipeline_result, profile_display_name, task_id,
                    ).replace("✅ 生成完成", "⚠️ 生成失败（部分结果）")
                    result_visible = True
                yield _build_yield(
                    error_value=f"生成失败：{error_msg}\n\n```\n{tb_short}\n```",
                    error_visible=True,
                    progress_value=_render_stage_panel_from_events(events_collected),
                    result_value=result_value,
                    result_visible=result_visible,
                    log_display_value=logs,
                    btn_interactive=True,
                )
                return

    except Exception as err:
        # generator 自身异常（理论不应该发生，run_pipeline_stream 内部已 try/except）
        tb_str = tb_mod.format_exc()
        yield _build_yield(
            error_value=(
                f"WebUI 异常：{type(err).__name__}: {err}\n\n```\n{tb_str[-1500:]}\n```"
            ),
            error_visible=True,
            btn_interactive=True,
        )
        return

    # 理论上不会走到这里；保险起见恢复按钮
    yield _build_yield(btn_interactive=True)


def reset_handler():
    """清空按钮。"""
    return (
        gr.update(value="", visible=False),                # 1. error_text (清空+隐藏)
        gr.update(value=""),                               # 2. textbox
        gr.update(value=None),                             # 3. file_upload
        gr.update(value=None),                             # 4. profile_dropdown
        gr.update(value=""),                               # 5. profile_desc
        gr.update(value=_render_stage_panel_initial()),    # 6. progress_panel
        gr.update(value=None, visible=False),              # 7. audio_player
        gr.update(value=None, visible=False),              # 8. download_btn
        gr.update(value="", visible=False),                # 9. result_info
        gr.update(value=""),                               # 10. log_display
        gr.update(interactive=True),                       # 11. generate_btn
    )


# ─── UI ──────────────────────────────────────────────────────────────────────


def create_ui() -> gr.Blocks:
    """构造 Gradio Blocks 界面。"""
    # 启动时扫描 profiles；过滤掉 blue_* （本服务只跑黄区内网）
    try:
        all_profiles = discover_profiles(_PROFILES_DIR)
    except Exception:
        all_profiles = []
    profiles = [
        p for p in all_profiles
        if not p.get("filename_stem", "").startswith("blue_")
        and p.get("region", "") != "blue"
    ]

    dropdown_choices = [(p["display_name"], p["path"]) for p in profiles]
    if dropdown_choices:
        default_profile_value = dropdown_choices[0][1]
        default_desc = ""
        for p in profiles:
            if p["path"] == default_profile_value:
                region = p.get("region", "")
                desc = p.get("description", "")
                default_desc = f"**[{region}]** {p['display_name']}\n\n{desc}"
                break
    else:
        default_profile_value = None
        default_desc = "_未发现可用 profile（请检查 src_next/profiles/）_"

    # 紧凑主题：减小默认 spacing / radius
    try:
        theme = gr.themes.Soft(spacing_size="sm", radius_size="md", text_size="md")
    except Exception:
        theme = gr.themes.Soft()

    with gr.Blocks(title="有声书生成器", theme=theme, css=CUSTOM_CSS) as app:
        with gr.Row():
            gr.Markdown("# 📻 有声书生成器")
            gr.Markdown(
                f"<div style='text-align:right; color:#888; font-size:13px; padding-top:14px'>"
                f"最多 {DEFAULT_CONCURRENCY} 个任务并发 · 队列上限 {DEFAULT_QUEUE_SIZE}"
                f"</div>",
                elem_classes=["concurrency-hint"],
            )

        with gr.Row():
            # 左栏：输入
            with gr.Column(scale=3):
                textbox = gr.Textbox(
                    label="故事文本",
                    placeholder=f"请输入故事文本（建议 ≤ 3000 字，硬上限 {MAX_TEXT_LENGTH} 字）...",
                    lines=12,
                    max_lines=15,
                )
                char_count = gr.Markdown(f"0 / {MAX_TEXT_LENGTH}")
                file_upload = gr.File(
                    label=f"或上传 TXT（UTF-8/GBK，≤ {MAX_TXT_FILE_SIZE_BYTES // 1024} KB）",
                    file_types=[".txt"],
                    file_count="single",
                )

            # 右栏：配置 + 进度
            with gr.Column(scale=2):
                profile_dropdown = gr.Dropdown(
                    label="Profile 选择",
                    choices=dropdown_choices,
                    value=default_profile_value,
                    interactive=True,
                )
                profile_description = gr.Markdown(default_desc)

                with gr.Row():
                    generate_btn = gr.Button("🎬 生成有声书", variant="primary", size="lg")
                    reset_btn = gr.Button("🔄 清空", size="lg")

                progress_panel = gr.Textbox(
                    label="生成进度",
                    value=_render_stage_panel_initial(),
                    lines=10,
                    interactive=False,
                    elem_classes=["stage-panel"],
                )
                error_text = gr.Markdown("", visible=False, elem_classes=["error-text"])

        # 结果区：紧凑布局，audio / download / result_info 上下排列
        with gr.Group():
            gr.Markdown("### 🎧 生成结果")
            result_info = gr.Markdown("", visible=False, elem_classes=["result-bar"])
            audio_player = gr.Audio(label="生成的音频", visible=False, show_download_button=False)
            download_btn = gr.DownloadButton(
                "💾 下载音频",
                visible=False,
                variant="primary",
                size="lg",
                elem_classes=["download-bar"],
            )

        # 日志区
        with gr.Group():
            gr.Markdown("### 📜 实时日志")
            log_display = gr.Textbox(label="", lines=8, interactive=False, max_lines=15)

        # 11 个输出组件顺序（必须和 generate_audiobook_handler / reset_handler 返回 tuple 一致）
        outputs = [
            error_text,              # 1
            textbox,                 # 2
            file_upload,             # 3
            profile_dropdown,        # 4
            profile_description,     # 5
            progress_panel,          # 6
            audio_player,            # 7
            download_btn,            # 8
            result_info,             # 9
            log_display,             # 10
            generate_btn,            # 11
        ]

        # 生成按钮：3 inputs（text / file_upload / profile_path）
        generate_btn.click(
            fn=generate_audiobook_handler,
            inputs=[textbox, file_upload, profile_dropdown],
            outputs=outputs,
        )

        reset_btn.click(
            fn=reset_handler,
            inputs=[],
            outputs=outputs,
        )

        # 文件上传事件
        file_upload.change(
            fn=on_file_upload,
            inputs=[file_upload],
            outputs=[textbox, file_upload, error_text],
        )

        # 字数统计
        textbox.change(
            fn=on_text_change,
            inputs=[textbox],
            outputs=[char_count],
        )

        # profile 切换
        profile_dropdown.change(
            fn=on_profile_change,
            inputs=[profile_dropdown],
            outputs=[profile_description],
        )

    return app


# ─── main ────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="src_next 有声书 WebUI")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"同时运行生成任务上限（默认 {DEFAULT_CONCURRENCY}）",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help=f"排队上限（默认 {DEFAULT_QUEUE_SIZE}）",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = create_ui()

    # Gradio 3.x 用 concurrency_count；4.x 用 default_concurrency_limit
    try:
        app.queue(
            concurrency_count=args.concurrency,
            max_size=args.queue_size,
        )
    except TypeError:
        try:
            app.queue(
                default_concurrency_limit=args.concurrency,
                max_size=args.queue_size,
            )
        except TypeError:
            app.queue(max_size=args.queue_size)

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=max(args.concurrency + 2, 10),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
