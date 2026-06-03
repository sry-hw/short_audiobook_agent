"""story_parser 模块测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from story_parser import parse_text


def _first_para(text):
    return parse_text(text)["paragraphs"][0]


# ── 段落拆分 ──────────────────────────────────────────────────


def test_split_blank_lines():
    """空行分段正确。"""
    text = "第一段。\n\n第二段。\n\n第三段。"
    result = parse_text(text)
    assert result["total_paragraphs"] == 3
    assert result["paragraphs"][0]["text"] == "第一段。"
    print("test_split_blank_lines 通过")


# ── 引号提取 ──────────────────────────────────────────────────


def test_extract_quotes():
    """引号内容正确提取。"""
    text = "小松鼠笑着说：“你好呀。”"
    parts = _first_para(text)["parts"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[0]["content"] == "小松鼠笑着说："
    assert parts[1]["type"] == "quoted"
    assert parts[1]["content"] == "你好呀。"
    print("test_extract_quotes 通过")


def test_no_quotes():
    """无引号段落 has_quotes=false，parts 只有一个 text。"""
    text = "先生讲得很详细，大家听得很认真。"
    para = _first_para(text)
    assert para["has_quotes"] is False
    assert len(para["parts"]) == 1
    assert para["parts"][0]["type"] == "text"
    print("test_no_quotes 通过")


def test_multiple_quotes():
    """一段多引号，parts 交替排列。"""
    text = "母亲担心：“可别来台风啊！”母亲念着：“只要不来台风就好。”"
    parts = _first_para(text)["parts"]
    types = [p["type"] for p in parts]
    assert types == ["text", "quoted", "text", "quoted"]
    assert parts[1]["content"] == "可别来台风啊！"
    assert parts[3]["content"] == "只要不来台风就好。"
    print("test_multiple_quotes 通过")


def test_empty_text_after_quote():
    """引号后无文本，不产生空 text part。"""
    text = "他说：“走吧。”"
    parts = _first_para(text)["parts"]
    assert len(parts) == 2
    assert parts[-1]["type"] == "quoted"
    print("test_empty_text_after_quote 通过")


# ── part_id 格式 ──────────────────────────────────────────────


def test_part_id_format():
    """part_id 格式为 p{段落号}_part{段内序号}。"""
    text = "他说：“好。”她问：“什么？”"
    para = _first_para(text)
    ids = [p["part_id"] for p in para["parts"]]
    assert ids == ["p1_part1", "p1_part2", "p1_part3", "p1_part4"]
    print("test_part_id_format 通过")


# ── 完整文件测试 ──────────────────────────────────────────────


def test_sample_story_01():
    """sample_story_01.txt 段落数和引号数正确。"""
    from text_loader import load_text

    file_path = Path(__file__).resolve().parent.parent / "input" / "sample_story_01.txt"
    text = load_text(str(file_path))
    result = parse_text(text)

    assert result["total_paragraphs"] == 9
    quoted_count = sum(
        1 for p in result["paragraphs"] for part in p["parts"] if part["type"] == "quoted"
    )
    assert quoted_count == 7
    print("test_sample_story_01 通过")


def test_indented_text():
    """不懂就要问.txt 9 个段落，引号位置正确。"""
    from text_loader import load_text

    file_path = Path(__file__).resolve().parent.parent / "input" / "不懂就要问.txt"
    text = load_text(str(file_path))
    result = parse_text(text)

    assert result["total_paragraphs"] == 9
    quoted_count = sum(
        1 for p in result["paragraphs"] for part in p["parts"] if part["type"] == "quoted"
    )
    assert quoted_count == 7
    print("test_indented_text 通过")


def test_guihua():
    """桂花雨.txt 引号和 parts 结构正确。"""
    from text_loader import load_text

    file_path = Path(__file__).resolve().parent.parent / "input" / "桂花雨.txt"
    text = load_text(str(file_path))
    result = parse_text(text)

    assert result["total_paragraphs"] == 8
    quoted_parts = [
        part
        for p in result["paragraphs"]
        for part in p["parts"]
        if part["type"] == "quoted"
    ]
    # 包含对白和引用（摇、摇花乐）
    assert len(quoted_parts) >= 6
    print("test_guihua 通过")


# ── 运行 ──────────────────────────────────────────────────────


if __name__ == "__main__":
    test_split_blank_lines()
    test_extract_quotes()
    test_no_quotes()
    test_multiple_quotes()
    test_empty_text_after_quote()
    test_part_id_format()
    test_sample_story_01()
    test_indented_text()
    test_guihua()
    print("\n全部测试通过")
