"""src_next/tts/

TTS 适配层：把 ``core.tts_instruction_builder`` 产出的通用 ``TTSInstruction``
转译为具体 TTS 后端（IndexTTS / CosyVoice / FishPro / Qwen TTS / mock）调用，
并生成分段 wav。

* 输入：``list[TTSInstruction]`` + ``VoicebankResult`` + ``output_dir``
* 输出：``list[AudioSegmentResult]``

本层只负责"通用指令 → 后端调用 → wav 文件"，不负责：
* 文本切分（已经是 Segment 层的产物）；
* 角色识别（analysis 层）；
* 导演指令（analysis 层）；
* 音色准备（voicebank 层）；
* 多段拼接（core/audio_merger.py）。

后端选择通过 ``src_next/profiles/*.yaml`` 配置 + ``registry.create_tts_adapter``
工厂决定，core 层只依赖 ``BaseTTSAdapter`` 抽象。
"""
