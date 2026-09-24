"""第 26 章测试 —— 测 mini-BPE 真正该有的性质。

跑法：

    ./.venv/bin/python -m pytest chapter_26/tests/ -q
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from after import (  # noqa: E402
    build_corpus,
    count_pairs,
    decode,
    encode,
    from_bytes,
    merge_pair,
    to_bytes,
    token_count,
    train_bpe,
)
from before import VOCABULARY, word_tokenize  # noqa: E402


@pytest.fixture(scope="module")
def merges():
    return train_bpe(build_corpus(), 100)


def test_合并的是当前最高频的相邻对():
    tokens = list("ababab")
    pairs = count_pairs(tokens)
    assert pairs[("a", "b")] == 3
    assert max(pairs.values()) == 3
    assert merge_pair(tokens, ("a", "b")) == ["ab", "ab", "ab"]


def test_一次合并只做一件事():
    learned = train_bpe("ababab", 1)
    assert learned == [("a", "b")]


def test_词表大小等于基础字符数加合并次数():
    corpus = build_corpus()
    num_merges = 50
    learned = train_bpe(corpus, num_merges)
    assert len(learned) == num_merges
    assert len(set(corpus)) + len(learned) == len(set(corpus)) + 50


def test_编码能原样还原(merges):
    for sentence in ["苹果很好吃", "苹果不好吃", "华为发布新平板",
                     "我昨天在商场看到苹果刚刚发布的新手机"]:
        tokens = encode(sentence, merges)
        assert decode(tokens) == sentence


def test_bpe_切出来的比字符级少(merges):
    sentence = "苹果很好吃"
    assert len(encode(sentence, merges)) < len(sentence)


def test_bpe_没有生词(merges):
    # 语料里连"不"都没有，BPE 照样切得开，不会吐 <UNK>
    tokens = encode("苹果不好吃", merges)
    assert "<UNK>" not in tokens
    assert decode(tokens) == "苹果不好吃"


def test_词级分词会遇到生词():
    tokens = word_tokenize("苹果不好吃")
    assert "<UNK>" in tokens              # 词表里没有"不"
    assert "苹果" in VOCABULARY


def test_学到的片段确实是语料里高频出现的(merges):
    learned = {left + right for left, right in merges}
    assert "苹果" in learned
    assert "发布" in learned


def test_合并次数越多序列越短():
    corpus = build_corpus()
    all_merges = train_bpe(corpus, 200)
    counts = [token_count(corpus, all_merges[:n]) for n in (0, 50, 200)]
    assert counts[0] > counts[1] > counts[2]


def test_字节版能编码任何文本():
    # 这些字全都不在训练语料里，字节版照样能编码、能还原
    for text in ["蘋果很好吃", "🍎很好吃", "hello 苹果"]:
        tokens = encode(text, [], unit=to_bytes)
        assert from_bytes(tokens) == text


def test_字节版训练后能把一个字的字节粘起来():
    corpus = build_corpus()
    byte_merges = train_bpe(corpus, 100, unit=to_bytes)
    text = "苹果很甜"
    before = token_count(text, [], unit=to_bytes)
    after = token_count(text, byte_merges, unit=to_bytes)
    assert before == 4 * 3          # 4 个字 = 12 个字节
    assert after < before


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
