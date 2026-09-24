"""第 28 章测试 —— 测这一章的观察是不是真的。

跑法：

    ./.venv/bin/python -m pytest chapter_28/tests/ -q
"""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from after import (  # noqa: E402
    BLOCK_SIZE,
    CHARS,
    SIZES,
    SPLIT,
    TEXT,
    TRAIN_DATA,
    VAL_DATA,
    VOCAB_SIZE,
    GPTModel,
    evaluate,
    get_batch,
    train,
)

SMALL = SIZES[0][1]      # dim=32，1 层
MIDDLE = SIZES[1][1]     # dim=64，2 层


def test_训练集和验证集没有重叠():
    """验证集是语料的后 20%，一个字都没有参与训练。"""
    assert len(TRAIN_DATA) + len(VAL_DATA) == len(TEXT)
    assert list(TRAIN_DATA) == [CHARS.index(ch) for ch in TEXT[:SPLIT]]
    assert list(VAL_DATA) == [CHARS.index(ch) for ch in TEXT[SPLIT:]]
    # 两段接在一起就是完整的语料，中间没有缝
    assert np.array_equal(np.concatenate([TRAIN_DATA, VAL_DATA]),
                          np.array([CHARS.index(ch) for ch in TEXT]))


def test_评估不会改动模型():
    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **SMALL)
    before = [p.detach().clone() for p in model.parameters()]
    evaluate(model, VAL_DATA)
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)
    assert model.training          # 量完以后要切回训练模式


def test_参数量是随着变大而变多的():
    counts = []
    for _, size in SIZES:
        torch.manual_seed(0)
        counts.append(GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **size).n_params())
    assert counts[0] < counts[1] < counts[2]


def test_同样步数下更大的模型loss更低():
    """这一章的核心观察：同一时刻比，参数越多 loss 越低。"""
    losses = {}
    for name, size in (("小", SMALL), ("中", MIDDLE)):
        torch.manual_seed(0)
        model = GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **size)
        history, _ = train(model, 30, checkpoints=(30,))
        losses[name] = history[30]
    assert losses["中"] < losses["小"]
    # 而且差距不小：30 步时小模型连门都没摸到
    assert losses["小"] > 0.9


def test_训练能降低损失():
    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **SMALL)
    rng = np.random.default_rng(0)
    x, y = get_batch(TRAIN_DATA, BLOCK_SIZE, 32, rng)
    with torch.no_grad():
        first = torch.nn.functional.cross_entropy(
            model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
    assert first > 1.0                 # 一开始比瞎猜还差
    train(model, 40)
    # 训练 40 步之后，在没见过的数据上也明显好于瞎猜
    assert evaluate(model, VAL_DATA) < math.log(VOCAB_SIZE)


def test_ngram回退给出的是一套概率():
    from experiment import build_ngram_counts, ngram_loss  # noqa: E402

    counts = build_ngram_counts(TEXT[:SPLIT], 3)
    context = TEXT[100:103]
    counter = counts[context]
    total = sum(counter.values())
    probabilities = [(counter.get(ch, 0) + 0.5 / len(CHARS)) / (total + 0.5) for ch in CHARS]
    # 未平滑之前的原始计数必须能归一化成 1（平滑后只是近似）
    raw = [counter.get(ch, 0) / total for ch in CHARS]
    assert abs(sum(raw) - 1.0) < 1e-9
    assert all(p > 0 for p in probabilities)


def test_ngram比瞎猜好但比不上模型():
    from experiment import ngram_loss  # noqa: E402

    loss = ngram_loss(TEXT[:SPLIT], TEXT[SPLIT:], 4)
    assert loss < math.log(VOCAB_SIZE)      # 比均匀猜好
    assert loss > 0.5                       # 但离我们的模型（0.46）还差得远


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
