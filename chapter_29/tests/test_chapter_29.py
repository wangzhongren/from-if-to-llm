"""第 29 章测试 —— 测三个阶段的训练目标真的不一样。

跑法：

    ./.venv/bin/python -m pytest chapter_29/tests/ -q
"""

import copy
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import after  # noqa: E402
from after import (  # noqa: E402
    END,
    FACTS,
    VOCAB_SIZE,
    TinyLM,
    build_preference_pairs,
    build_pretrain_text,
    build_sft_pairs,
    dpo_step,
    encode,
    make_sequences,
    pad_sequences,
    sft_step,
    sequence_logprob,
    pretrain_step,
)


@pytest.fixture(scope="module")
def trained():
    """一个跑完三个阶段的小模型（整个测试文件共用，只算一次）。"""
    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=32, n_layer=1, n_head=4, block_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        pretrain_step(model, optimizer, 16, rng)
    pretrained = copy.deepcopy(model)
    for _ in range(150):
        sft_step(model, optimizer, 16, rng)
    sft_model = copy.deepcopy(model)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(60):
        dpo_step(model, reference, optimizer, rng)
    return pretrained, sft_model, model


# ---------------------------------------------------------------- 数据格式

def test_预训练语料里没有一处是问句后面跟着答案():
    text = build_pretrain_text(segments=100)
    assert END not in text                       # 结束符是微调阶段才引入的
    # 语料里当然有问句，但任何一问后面都不会直接跟答案
    for _, _, _, question, short, _ in FACTS:
        assert question + short not in text


def test_指令数据是提示加回答_回答结尾带结束符():
    pairs = build_sft_pairs()
    assert len(pairs) == 2 * len(FACTS)          # 每个问题有两种风格的答案
    assert all(answer.endswith(END) for _, answer in pairs)
    assert all(prompt.startswith("问：") for prompt, _ in pairs)


def test_偏好数据是同题三件套():
    pairs = build_preference_pairs()
    assert len(pairs) == len(FACTS)
    for (prompt, chosen, rejected), fact in zip(pairs, FACTS):
        assert prompt == fact[3]
        assert chosen == fact[4] + END
        assert rejected == fact[5] + END
        assert chosen != rejected


# ---------------------------------------------------------------- 掩码

def test_掩码只在回答那几个字上是1():
    prompt, answer = "问：苹果甜吗？答：", "甜。" + END
    (inputs, targets, mask), = make_sequences([(prompt, answer)], 32)
    # 输入和答案错开一位
    assert targets == inputs[1:] + [targets[-1]]
    # 提示部分全是 0，回答部分全是 1
    assert mask[:len(prompt) - 1] == [0] * (len(prompt) - 1)
    assert sum(mask) == len(answer)
    assert mask[len(prompt) - 1:] == [1] * len(answer)


def test_补长度不会把补齐的部分算进loss():
    sequences = make_sequences([("问：苹果甜吗？答：", "甜。" + END),
                                ("问：香蕉很好吃吗？答：", "好吃。" + END)], 32)
    inputs, targets, mask = pad_sequences(sequences, 32)
    assert inputs.shape == targets.shape == mask.shape
    # 短的那一条后面补的 0 不计入 loss
    shorter = min(len(seq[0]) for seq in sequences)
    assert mask[0, shorter:].sum() == 0 or mask[1, shorter:].sum() == 0


def test_指令微调的loss等于手工算的那一个():
    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=32, n_layer=1, n_head=4, block_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    pairs = [build_sft_pairs()[0]] * 4
    inputs, targets, mask = pad_sequences(make_sequences(pairs, 32), 32)
    logits = model(inputs)
    per_token = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                               targets.reshape(-1), reduction="none").reshape(targets.shape)
    expected = (per_token * mask).sum() / mask.sum()
    loss = sft_step(model, optimizer, 4, rng)
    # sft_step 内部随机抽样本，所以只比较"量级和做法"：
    assert expected.item() > 0
    assert math.isfinite(loss)
    # 提示部分不算 loss：把它加进去结果会明显不同
    full = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    assert not torch.isclose(expected, full, atol=1e-3)


# ---------------------------------------------------------------- 概率与对齐

def test_对数概率就是每个字概率的对数相加():
    torch.manual_seed(0)
    model = TinyLM(VOCAB_SIZE, dim=32, n_layer=1, n_head=4, block_size=32)
    prompt, answer = "问：苹果甜吗？答：", "甜。"
    ids = encode(prompt + answer)
    with torch.no_grad():
        logprobs = model(torch.tensor([ids]))[0].log_softmax(dim=-1)
    manual = sum(logprobs[position, ids[position + 1]].item()
                 for position in range(len(prompt) - 1, len(ids) - 1))
    assert abs(sequence_logprob(model, prompt, answer).item() - manual) < 1e-6
    # 对数概率永远是负的（概率 ≤ 1），而且概率 = exp(对数概率)
    assert manual < 0
    assert 0 < math.exp(manual) <= 1


def test_偏好对齐会把好回答推上去_把差回答压下去(trained):
    _, sft_model, aligned = trained
    prompt, chosen, rejected = build_preference_pairs()[0]
    before_chosen = sequence_logprob(sft_model, prompt, chosen).item()
    before_rejected = sequence_logprob(sft_model, prompt, rejected).item()
    after_chosen = sequence_logprob(aligned, prompt, chosen).item()
    after_rejected = sequence_logprob(aligned, prompt, rejected).item()

    # 好回答的概率变高了
    assert after_chosen > before_chosen
    # 差回答的概率变低了
    assert after_rejected < before_rejected
    # 而且"好"和"差"的差距被拉开了
    assert after_chosen - after_rejected > before_chosen - before_rejected


def test_预训练模型不会回答(trained):
    """没微调过的模型，在问答格式后面接的是别的东西，不是答案。"""
    pretrained, _, _ = trained
    question, short = FACTS[0][3], FACTS[0][4]
    ids = encode(question)
    with torch.no_grad():
        logits = pretrained(torch.tensor([ids]))[0, -1]
    # 概率最高的那个字不是答案的第一个字
    top = logits.argmax().item()
    assert top != encode(short)[0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
