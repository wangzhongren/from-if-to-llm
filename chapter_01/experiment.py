"""
这个文件让读者亲眼看到两件事：

1. 不拆句子，程序只能认"一模一样"的字符串——多一个"了"就不认识了；
2. 用词表把句子拆开之后，不同写法的句子之间出现了共享的词，
   "这两句像不像"第一次有了一个可以数的东西。

实验一是旧方法，实验二和实验三是把句子拆开之后的样子。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 别的章节也可能有同名的 before.py / after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("before", None)
sys.modules.pop("after", None)

from before import KNOWN_SENTENCES, REWRITTEN_SENTENCES, has_seen
from after import split_sentence, to_ids, UNKNOWN_WORD


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了对齐好看。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


def shared_words(sentence_a, sentence_b):
    """
    两句拆开之后，一共有多少个共同的词。

    注意 <UNK> 不算：它只是一个"这儿的字我不认识"的占位符，
    两个句子里各有一个 <UNK>，并不代表它们有什么共同之处。
    """
    words_a = [w for w in split_sentence(sentence_a) if w != UNKNOWN_WORD]
    words_b = [w for w in split_sentence(sentence_b) if w != UNKNOWN_WORD]
    remaining = list(words_b)
    count = 0
    for word in words_a:
        if word in remaining:
            remaining.remove(word)
            count += 1
    return count


def main():
    # ---------------- 实验一 ----------------
    print("=" * 60)
    print("实验一：不拆句子，只比字符串")
    print("=" * 60)
    hit = sum(1 for s in KNOWN_SENTENCES if has_seen(s))
    rewrite_hit = sum(1 for s in REWRITTEN_SENTENCES if has_seen(s))
    print(f"  语料里的 10 句原话     命中 {hit}/{len(KNOWN_SENTENCES)}")
    print(f"  换个写法的 5 句话      命中 {rewrite_hit}/{len(REWRITTEN_SENTENCES)}")
    print()
    print("  换个写法的这 5 句，逐句看：")
    for sentence in REWRITTEN_SENTENCES:
        result = "认识" if has_seen(sentence) else "不认识"
        print(f"    {pad(sentence, 24)} {result}")
    print()

    # ---------------- 实验二 ----------------
    print("=" * 60)
    print("实验二：拆开之后，一句话变成了什么")
    print("=" * 60)
    examples = [
        "苹果发布新手机",
        "苹果发布了新手机",
        "苹果芯片很强",
        "这个苹果真甜",
        "苹果配香蕉",
    ]
    for sentence in examples:
        words = split_sentence(sentence)
        print(f"  {sentence}")
        print(f"      {' | '.join(words)}")
        print(f"      {to_ids(words)}")
    print()
    print(f"  说明：词表里没有的块（'了'、'强'、空格……）都换成了 {UNKNOWN_WORD}。")
    print()

    # ---------------- 实验三 ----------------
    print("=" * 60)
    print("实验三：拆开之后，'像不像'第一次可以数出来")
    print("=" * 60)
    sample = REWRITTEN_SENTENCES[0]
    print(f"  拿「{sample}」和语料里的每一句比，数一数共享了几个词：")
    print()
    sample_words = split_sentence(sample)
    for known in KNOWN_SENTENCES:
        count = shared_words(sample, known)
        known_words = split_sentence(known)
        common = [w for w in sample_words if w in known_words and w != UNKNOWN_WORD]
        print(f"    {pad(known, 18)} 共享 {count} 个：{' '.join(common)}")
    print()

    print("  5 句「换个写法」的最像的一句：")
    for sentence in REWRITTEN_SENTENCES:
        counts = [(shared_words(sentence, known), known) for known in KNOWN_SENTENCES]
        best_count, best_sentence = max(counts)
        print(f"    {pad(sentence, 24)} 最像 -> {pad(best_sentence, 16)} 共享 {best_count} 个词")
    print()
    print("-" * 60)
    print("第一句和语料里的第一句共享 4 个词——多出来的那个'了'并不妨碍它们")
    print("看起来像同一句话。这就是把句子拆开换来的东西。")


if __name__ == "__main__":
    main()
