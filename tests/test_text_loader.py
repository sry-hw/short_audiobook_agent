"""text_loader 模块的简单测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_loader import load_text


def test_load_sample_story():
    """能读取 input/sample_story_01.txt，且返回非空字符串。"""
    file_path = Path(__file__).resolve().parent.parent / "input" / "sample_story_01.txt"
    text = load_text(str(file_path))
    assert isinstance(text, str)
    assert len(text) > 0


def test_file_not_found():
    """文件不存在时抛出 FileNotFoundError。"""
    try:
        load_text("nonexistent_file.txt")
        assert False, "应该抛出 FileNotFoundError"
    except FileNotFoundError as e:
        assert "文件不存在" in str(e)


if __name__ == "__main__":
    test_load_sample_story()
    print("test_load_sample_story 通过")
    test_file_not_found()
    print("test_file_not_found 通过")
    print("全部测试通过")
