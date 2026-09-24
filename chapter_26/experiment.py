"""第 26 章 experiment.py —— 四种分词法摆在一起比。

  一、同一批句子，四种分词法各切出什么（含生词情况）
  二、BPE 的合并次数 = 词表大小 = 序列长度，这条线怎么权衡
  三、BPE 自己"发现"了多少第 1 章那张词表里的词

跑法：

    ./.venv/bin/python chapter_26/experiment.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    build_corpus,
    encode,
    to_bytes,
    train_bpe,
)
from before import (  # noqa: E402
    UNK,
    VOCABULARY,
    char_tokenize,
    count_ngrams,
    word_tokenize,
)

TEST_SENTENCES = [
    "苹果不好吃",
    "苹果很甜",
    "华为发布新平板",
    "我昨天在商场看到苹果刚刚发布的新手机",
]

NUM_MERGES = 200


def pad(text, width):
    """中文字符占两个格子，直接 f-string 对齐会歪。"""
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def show_byte_tokens(tokens):
    """字节版切出来的 token 可能只是某个字的一半，单独一个不一定能显示。"""
    return [t.decode("utf-8", errors="replace") for t in tokens]


def experiment_one(corpus, char_merges, byte_merges):
    print("=" * 72)
    print("一、四种分词法，切同一批句子")
    print("=" * 72)
    char_vocab_size = count_ngrams(corpus, 1)
    print(f"字符级词表：{char_vocab_size} 个（语料里出现过的字）")
    print(f"词级词表：  {len(VOCABULARY)} 个（第 1 章手写的那张表）")
    print(f"BPE 词表：  {char_vocab_size} + {NUM_MERGES} = {char_vocab_size + NUM_MERGES} 个"
          f"（合并出来的新 token 全部进词表）")
    print(f"字节版 BPE：256 + {NUM_MERGES} = {256 + NUM_MERGES} 个")
    print()

    for sentence in TEST_SENTENCES:
        print(f"{sentence}")
        char_tokens = char_tokenize(sentence)
        word_tokens = word_tokenize(sentence)
        our_tokens = encode(sentence, char_merges)
        byte_tokens = show_byte_tokens(encode(sentence, byte_merges, unit=to_bytes))
        mark = f"   <- {word_tokens.count(UNK)} 个生词" if UNK in word_tokens else ""
        print(f"  字符级   {len(char_tokens):>2} 个  {char_tokens}")
        print(f"  词级     {len(word_tokens):>2} 个  {word_tokens}{mark}")
        print(f"  BPE      {len(our_tokens):>2} 个  {our_tokens}")
        print(f"  字节版   {len(byte_tokens):>2} 个  {byte_tokens}")
        print()
    print("（字节版里的 '�' 是半个字的字节：那个字在训练语料里没出现过，")
    print("  所以 BPE 还没学会把它粘起来。字节版永远切得开，但可能切得难看。）")
    print()

    print("四种分词法的体检报告：")
    print()
    print(pad("方案", 16) + pad("词表大小", 12) + pad("token 数", 12) + "生词")
    print("-" * 72)
    rows = [
        ("字符级", char_vocab_size, len(char_tokenize("苹果不好吃")), "没有生词，但序列长"),
        ("词级", len(VOCABULARY), len(word_tokenize("苹果不好吃")), "有 <UNK>"),
        ("BPE（字符起步）", char_vocab_size + NUM_MERGES,
         len(encode("苹果不好吃", char_merges)), "没有生词"),
        ("BPE（字节起步）", 256 + NUM_MERGES,
         len(encode("苹果不好吃", byte_merges, unit=to_bytes)), "没有生词，任何文本都行"),
    ]
    for name, vocab_size, tokens, note in rows:
        print(pad(name, 16) + pad(str(vocab_size), 12) + pad(str(tokens), 12) + note)
    print()


def experiment_two(corpus, sentence="我昨天在商场看到苹果刚刚发布的新手机"):
    print("=" * 72)
    print("二、合并次数 = 词表大小 = 序列长度")
    print("=" * 72)
    print("BPE 的全部旋钮只有一个：合并多少次。")
    print()
    print(pad("合并次数", 12) + pad("词表大小", 12) + pad("语料总 token 数", 18) + "这句话几个 token")
    print("-" * 72)
    base = count_ngrams(corpus, 1)
    all_merges = train_bpe(corpus, 500)
    for count in (0, 50, 100, 200, 500):
        merges = all_merges[:count]
        total = len(encode(corpus, merges))
        this = len(encode(sentence, merges))
        print(pad(str(count), 12) + pad(str(base + count), 12) + pad(f"{total:,}", 18) + str(this))
    print()
    print("合并得越多：词表越大、序列越短。")
    print("词表大 = 输出层的矩阵大；序列短 = 模型要算的位置少。这就是分词要权衡的东西。")
    print()


def experiment_three(corpus, merges):
    print("=" * 72)
    print("三、BPE 自己'发现'了多少第 1 章的词")
    print("=" * 72)
    # 把所有合并出来的 token 收集起来（它们就是"从数据里长出来的词"）
    learned = set(corpus)          # 一开始是字符
    for left, right in merges:
        learned.add(left + right)
    words = [w for w in VOCABULARY if len(w) > 1]
    hit = [w for w in words if w in learned]
    print(f"第 1 章那张词表里，长度大于 1 的词有 {len(words)} 个：{'  '.join(words)}")
    print(f"其中被 BPE 自己合并出来的：{len(hit)} 个 -> {'  '.join(hit)}")
    print()
    print("没有人告诉 BPE 中文的词长什么样。它只知道'哪两个 token 老挨在一起'。")
    print("结果它把'苹果''发布''手机''好吃'这些片段拼了出来 —— 因为它们确实老挨在一起。")
    print()


if __name__ == "__main__":
    corpus = build_corpus()
    char_merges = train_bpe(corpus, NUM_MERGES)
    byte_merges = train_bpe(corpus, NUM_MERGES, unit=to_bytes)
    experiment_one(corpus, char_merges, byte_merges)
    experiment_two(corpus)
    experiment_three(corpus, char_merges)
