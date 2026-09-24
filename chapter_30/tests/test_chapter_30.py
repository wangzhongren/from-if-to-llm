"""第 30 章测试 —— 测"两种做法"的差别是真的。

跑法：

    ./.venv/bin/python -m pytest chapter_30/tests/ -q
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import after  # noqa: E402
import before  # noqa: E402
from after import label_probabilities  # noqa: E402


@pytest.fixture(scope="module")
def model():
    """训练一个分类模型（整个测试文件共用一次）。"""
    torch.manual_seed(0)
    net = after.TinyLM(after.VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=48)
    after.train(net, steps=600)       # 和 after.py 里一样的步数
    return net


# ---------------------------------------------------------------- 第 2 章那一半

def test_if_else分类器照着规则工作():
    for sentence in ["苹果发布新手机", "华为发布新电脑"]:
        assert before.classify(before.split_words(sentence)) == "科技"
    for sentence in ["苹果很好吃", "香蕉很好吃"]:
        assert before.classify(before.split_words(sentence)) == "食品"


def test_一个关键词都没碰上时会掉进默认分支():
    words = before.split_words("床前明月光")
    assert words == []                          # 一个词都切不出来
    assert before.classify(words) == "食品"     # 然后掉进那句"先猜食品"


def test_规则的知识就是那张词表():
    """删掉一个词，靠它判断的句子立刻掉分 —— 规则不会自己补回来。"""
    sentence = "小米直播带货"
    assert before.classify(before.split_words(sentence)) == "科技"

    original = list(before.TECH_WORDS)
    before.TECH_WORDS = [w for w in original if w != "小米"]
    try:
        assert before.classify(before.split_words(sentence)) == "食品"   # 掉进默认分支
    finally:
        before.TECH_WORDS = original


# ---------------------------------------------------------------- 第 30 章那一半

def test_模型给出的是一个概率(model):
    probabilities = label_probabilities(model, "苹果很甜")
    assert set(probabilities) == {"科技", "食品"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    for value in probabilities.values():
        assert 0.0 <= value <= 1.0


def test_模型不用一行规则就能分类(model):
    """训练集里从来没有出现过'直播带货''平板'这两个说法，它也能判断。"""
    for sentence, expected in [("小米直播带货", "科技"), ("华为发布新平板", "科技"),
                               ("苹果很甜", "食品"), ("香蕉做成派很好吃", "食品")]:
        probabilities = label_probabilities(model, sentence)
        assert max(probabilities, key=probabilities.get) == expected
        assert probabilities[expected] > 0.8


def test_模型和规则在一句话上会给出不同的答案(model):
    """「苹果手机很甜」：规则按词序判成科技，模型按'甜'判成食品。

    这不是谁对谁错 —— 这句话本来就有歧义。
    但它说明两种做法的"依据"完全不一样：
    规则看的是关键词出现在第几个，模型看的是整句话合起来像什么。
    """
    sentence = "苹果手机很甜"
    assert before.classify(before.split_words(sentence)) == "科技"
    probabilities = label_probabilities(model, sentence)
    assert max(probabilities, key=probabilities.get) == "食品"


def test_两种做法在这一批句子上打成平手(model):
    """这一章最重要的一张成绩单：打平。"""
    rule_hits = 0
    model_hits = 0
    total = 0
    for sentence, label in after.TEST_SET:
        if label == "说不清":
            continue
        total += 1
        rule_hits += before.classify(before.split_words(sentence)) == label
        probabilities = label_probabilities(model, sentence)
        model_hits += max(probabilities, key=probabilities.get) == label
    assert rule_hits == total
    assert model_hits == total


def test_全书的最后几行代码能跑(model):
    text = after.continue_text(model, "床前明月光", n_new=12, seed=0)
    assert text.startswith("床前明月光")
    assert len(text) == 5 + 12


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
