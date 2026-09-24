"""
这个文件让读者亲眼看到"手写规则"是怎么一步步变成麻烦的。

三组实验：
1. 补丁史：规则不是想出来的，是补出来的——每补一条，正确率涨一点；
2. 换顺序：一条规则都不改，只把两条规则调换位置，正确率就掉下来；
3. 出题外：拿语料之外的句子去考它，看它什么时候开始答不上来。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 别的章节也可能有同名的 before.py / after.py，先清掉缓存
sys.modules.pop("before", None)
sys.modules.pop("after", None)

from after import CORPUS, judge, pad, split_sentence


# ---------- 三个历史版本 ----------

def judge_v1(words):
    """第 1 遍：只写了一条规则。"""
    if "手机" in words:
        return "科技"
    return "不知道"


def judge_v2(words):
    """第 2 遍：发现科技句的错法不止一种，补上几个词。"""
    if "手机" in words:
        return "科技"
    if "芯片" in words:
        return "科技"
    if "电脑" in words:
        return "科技"
    if "华为" in words:
        return "科技"
    if "小米" in words:
        return "科技"
    if "发布" in words:
        return "科技"
    return "不知道"


def judge_v3(words):
    """第 3 遍：食品那边一句都没答上来，补上食品词。"""
    if "手机" in words:
        return "科技"
    if "芯片" in words:
        return "科技"
    if "电脑" in words:
        return "科技"
    if "华为" in words:
        return "科技"
    if "小米" in words:
        return "科技"
    if "发布" in words:
        return "科技"
    if "好吃" in words:
        return "食品"
    if "很" in words:
        return "食品"
    if "甜" in words:
        return "食品"
    if "香蕉" in words:
        return "食品"
    if "派" in words:
        return "食品"
    if "做成" in words:
        return "食品"
    return "不知道"


def judge_rules_reordered(words):
    """
    和第 2 章 after.py 里的规则一模一样，只调换了两条规则的位置：
    "苹果"那条从最后挪到了最前面，"很"那条从"好吃"后面挪到了它前面。
    规则一条没多、一条没少。
    """
    if "苹果" in words:          # 本来在最后，是兜底
        return "科技"
    if "很" in words:            # 本来排在"芯片"后面
        return "食品"
    if "手机" in words:
        return "科技"
    if "芯片" in words:
        return "科技"
    if "电脑" in words:
        return "科技"
    if "华为" in words:
        return "科技"
    if "小米" in words:
        return "科技"
    if "发布" in words:
        return "科技"
    if "好吃" in words:
        return "食品"
    if "甜" in words:
        return "食品"
    if "香蕉" in words:
        return "食品"
    if "派" in words:
        return "食品"
    if "做成" in words:
        return "食品"
    return "不知道"


# ---------- 语料之外的句子 ----------

NEW_SENTENCES = [
    ("苹果好香", "食品"),
    ("电脑好贵", "科技"),
    ("香蕉派很好吃", "食品"),
    ("苹果配奶油", "食品"),
    ("机器人发布新手机", "科技"),
]


def score(judge_function, sentences):
    """跑一遍，返回 (答对的句数, 每一句的结果)。"""
    results = []
    for sentence, label in sentences:
        answer = judge_function(split_sentence(sentence))
        results.append((sentence, label, answer))
    right = sum(1 for _, label, answer in results if answer == label)
    return right, results


def main():
    # ---------------- 实验一 ----------------
    print("=" * 66)
    print("实验一：补丁史")
    print("=" * 66)
    versions = [
        ("第 1 版  只有一条规则", judge_v1),
        ("第 2 版  补上科技词", judge_v2),
        ("第 3 版  补上食品词", judge_v3),
    ]
    for name, judge_function in versions:
        right, results = score(judge_function, CORPUS)
        wrong = [sentence for sentence, label, answer in results if answer != label]
        shown = "、".join(wrong[:3]) + ("……" if len(wrong) > 3 else "")
        print(f"  {pad(name, 22)} 正确 {right:>2}/10   答错 {len(wrong)} 句：{shown or '无'}")
    print()
    print("  补到第 3 版才全对。注意每一版之间隔着的不是'思考'，是'跑一遍看看错在哪'。")
    print()

    # ---------------- 实验二 ----------------
    print("=" * 66)
    print("实验二：一条规则都不改，只调换两条规则的位置")
    print("=" * 66)
    for name, judge_function in [("after.py 里的顺序", judge), ("调换两条之后", judge_rules_reordered)]:
        right, results = score(judge_function, CORPUS)
        print(f"  {pad(name, 22)} 正确 {right:>2}/10")
    print()
    print("  调换之后答错的句子：")
    _, results = score(judge_rules_reordered, CORPUS)
    for sentence, label, answer in results:
        if answer != label:
            print(f"    {pad(sentence, 18)} 答：{pad(answer, 6)} 真实：{label}")
    print()
    print("  规则一条没多、一条没少，只是把'苹果'和'很'这两条挪到了前面。")
    print()

    # ---------------- 实验三 ----------------
    print("=" * 66)
    print("实验三：拿语料之外的句子考它")
    print("=" * 66)
    right, results = score(judge, NEW_SENTENCES)
    for sentence, label, answer in results:
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 20)} 答：{pad(answer, 8)} 真实：{label}")
    print(f"  正确 {right}/{len(NEW_SENTENCES)}")
    print()
    print("  答错的两句，错法完全不一样：")
    print("    '苹果好香'  —— '香'不在词表里，规则根本看不见它；")
    print("    '苹果配奶油' —— 只剩'苹果'，兜底规则猜了'科技'。")
    print()
    print("-" * 66)
    print("规则是我们一条条补出来的，所以它只会处理我们补过的那几种情况。")


if __name__ == "__main__":
    main()
