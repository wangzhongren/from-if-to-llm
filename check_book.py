#!/usr/bin/env python
"""《从 If 到 LLM》完稿检查器。

跑一遍，看这本书有没有违反自己的规矩：

    ./.venv/bin/python check_book.py

检查三件事：

1. 目录结构 —— 每章该有的文件在不在
2. README 结构 —— 八个小节标题是不是一字不差
3. 术语纪律 —— 有没有提前使用后面章节才解释的概念

术语检查只报"可疑"，不报"错误"。有些词在口语里出现是合理的，
需要人来判断。真正硬性的错误（文件缺失、小节缺失）才会让脚本退出码非零。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = ["README.md", "before.py", "after.py", "experiment.py", "exercises.md"]

SECTIONS = [
    "## 本章的问题",
    "## 最简单的尝试",
    "## 实验",
    "## 新机制",
    "## Python 实现",
    "## 它解决了什么",
    "## 它还解决不了什么",
    "## 练习",
]

# 术语 -> 最早可以在第几章出现
FIRST_ALLOWED = {
    "机器学习": 4,
    "分类器": 4,
    "梯度": 5,
    "损失函数": 5,
    "交叉熵": 5,
    "softmax": 5,
    "激活函数": 8,
    "自动求导": 9,
    "embedding": 10,
    "词向量": 10,
    "上下文向量": 12,
    "Attention": 13,
    "注意力": 13,
    "Query": 14,
    "QKV": 14,
    "多头": 15,
    "位置编码": 16,
    "残差": 17,
    "LayerNorm": 18,
    "Transformer": 19,
    "语言模型": 20,
    "因果掩码": 21,
    "自回归": 23,
    "tokenizer": 20,   # 第 1 章刻意不命名，但第 20 章起作为通称可用（见 WRITING_SPEC 第 8 节）
    "BPE": 26,
    "微调": 29,
}

# 这些词太常见，只检查 README.md，不检查代码注释
README_ONLY = {"分类器", "机器学习", "注意力", "上下文向量", "语言模型", "自动求导"}


def chapter_dirs():
    out = []
    for name in sorted(os.listdir(ROOT)):
        m = re.fullmatch(r"chapter_(\d+)", name)
        if m and os.path.isdir(os.path.join(ROOT, name)):
            out.append((int(m.group(1)), name))
    return out


def check_structure(num, ch):
    errors, warns = [], []
    base = os.path.join(ROOT, ch)

    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(base, f)):
            errors.append(f"缺少 {ch}/{f}")

    tests = os.path.join(base, "tests")
    if not os.path.isdir(tests):
        errors.append(f"缺少 {ch}/tests/")
    elif not any(t.startswith("test_") and t.endswith(".py") for t in os.listdir(tests)):
        errors.append(f"{ch}/tests/ 里没有 test_*.py")

    # README 八节
    readme = os.path.join(base, "README.md")
    if os.path.isfile(readme):
        text = open(readme, encoding="utf-8").read()
        for s in SECTIONS:
            if s not in text:
                errors.append(f"{ch}/README.md 缺少小节：{s}")
        # 小节顺序
        idx = [text.find(s) for s in SECTIONS if s in text]
        if idx != sorted(idx):
            errors.append(f"{ch}/README.md 小节顺序不对")

    return errors, warns


def check_terms(num, ch):
    """术语纪律：扫这一章的 README 和 .py，看有没有提前用后面的概念。"""
    warns = []
    base = os.path.join(ROOT, ch)
    files = []
    for dirpath, _, names in os.walk(base):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n.endswith((".md", ".py")):
                files.append(os.path.join(dirpath, n))

    hits = {}
    for path in files:
        is_readme = path.endswith("README.md")
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, 1):
            for term, allowed in FIRST_ALLOWED.items():
                if num >= allowed:
                    continue
                if term in README_ONLY and not is_readme:
                    continue
                if term.lower() in line.lower():
                    hits.setdefault(term, []).append(
                        (os.path.relpath(path, ROOT), lineno, line.strip()[:70])
                    )

    for term, places in hits.items():
        warns.append(
            f"{ch} 用了「{term}」（最早应在第 {FIRST_ALLOWED[term]} 章出现），"
            f"共 {len(places)} 处，例如 {places[0][0]}:{places[0][1]}"
        )
    return warns


def main():
    chapters = chapter_dirs()
    if not chapters:
        print("没找到任何 chapter_XX 目录")
        return 1

    missing_nums = [n for n in range(1, 31) if n not in dict(chapters)]
    all_errors, all_warns = [], []

    print("=" * 70)
    print(f" 检查 {len(chapters)} 章")
    print("=" * 70)
    print()

    for num, ch in chapters:
        errors, _ = check_structure(num, ch)
        warns = check_terms(num, ch)
        all_errors += errors
        all_warns += warns

        status = "OK" if not errors else f"{len(errors)} 个问题"
        print(f"  第 {num:2d} 章  {status:<14} 术语可疑 {len(warns)} 处")

    print()
    print("=" * 70)
    print(" 结构问题（必须修）")
    print("=" * 70)
    if all_errors:
        for e in all_errors:
            print(f"  ✗ {e}")
    else:
        print("  无")

    print()
    print("=" * 70)
    print(" 术语纪律（需要人判断）")
    print("=" * 70)
    if all_warns:
        for w in all_warns:
            print(f"  ? {w}")
    else:
        print("  无")

    if missing_nums:
        print()
        print(f"  还没写的章节：{missing_nums}")

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
