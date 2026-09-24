"""第 23 章的测试：三个采样策略各自做对了没有。

跑法：

    ./.venv/bin/python -m pytest chapter_23/tests/ -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from after import (  # noqa: E402
    CONTEXT,
    CORPUS,
    VOCAB_SIZE,
    CharTokenizer,
    TinyLanguageModel,
    generate,
    pick_next_token,
)

TOKENIZER = CharTokenizer(CORPUS)

# 下面这几个测试测的是"采样策略"和"生成循环"，跟模型训得好不好没关系，
# 所以用一个没训练过的小模型就够了——测试要跑得快。
SMALL_MODEL = TinyLanguageModel(VOCAB_SIZE, dim=16, layers=1, context=CONTEXT)


def test_greedy_always_picks_the_same_token():
    """top_k=1 就是贪心：不管抽多少次，都是同一个字符。"""
    probabilities = np.array([0.5, 0.3, 0.15, 0.05])
    picks = {pick_next_token(probabilities, top_k=1, rng=np.random.default_rng(seed))
             for seed in range(50)}
    assert picks == {0}


def test_greedy_ignores_temperature():
    """贪心不看温度：它只数大小，概率被拧成什么样都不影响第一名。"""
    probabilities = np.array([0.5, 0.3, 0.15, 0.05])
    picks = {pick_next_token(probabilities, temperature=t, top_k=1,
                             rng=np.random.default_rng(0))
             for t in (0.1, 1.0, 10.0)}
    assert picks == {0}


def test_top_k_never_picks_outside_the_top_k():
    """top-k 是一个硬门槛：k 以外的候选永远不会被抽到。"""
    probabilities = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    for seed in range(200):
        picked = pick_next_token(probabilities, temperature=5.0, top_k=2,
                                 rng=np.random.default_rng(seed))
        assert picked in (0, 1)


def test_a_low_temperature_sharpens_the_distribution():
    """温度低 -> 更集中。同一个分布拧低温度之后，头名的机会更大。"""
    probabilities = np.array([0.4, 0.3, 0.2, 0.1])
    sharp = np.array([pick_next_token(probabilities, temperature=0.2, rng=np.random.default_rng(s))
                      for s in range(400)])
    flat = np.array([pick_next_token(probabilities, temperature=5.0, rng=np.random.default_rng(s))
                     for s in range(400)])
    assert (sharp == 0).mean() > (flat == 0).mean()


def test_a_high_temperature_reaches_rare_tokens():
    """温度高 -> 尾巴上的候选也有机会。这是"敢冒险"的意思。"""
    probabilities = np.array([0.97, 0.01, 0.01, 0.01])
    low = {pick_next_token(probabilities, temperature=0.5, rng=np.random.default_rng(s))
           for s in range(200)}
    high = {pick_next_token(probabilities, temperature=5.0, rng=np.random.default_rng(s))
            for s in range(200)}
    assert low == {0}
    assert len(high) > 1


def test_temperature_zero_is_rejected_loudly():
    """温度必须大于 0。我们不偷偷帮读者兜底。"""
    probabilities = np.array([0.5, 0.5])
    try:
        pick_next_token(probabilities, temperature=0.0, rng=np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("temperature=0 应该直接报错，而不是悄悄换成别的行为")


def test_generation_starts_with_the_prompt():
    """自回归生成的第一件事：把开头原样留在最前面。"""
    prompt = "床前明月"
    text = generate(SMALL_MODEL, TOKENIZER, prompt, length=5, seed=0)
    assert text.startswith(prompt)


def test_generation_is_deterministic_given_a_seed():
    """同一个种子，结果必须一模一样；换个种子，结果应该不一样。"""
    first = generate(SMALL_MODEL, TOKENIZER, "床前", 12, temperature=4.0, seed=0)
    again = generate(SMALL_MODEL, TOKENIZER, "床前", 12, temperature=4.0, seed=0)
    other = generate(SMALL_MODEL, TOKENIZER, "床前", 12, temperature=4.0, seed=1)
    assert first == again
    assert first != other


def test_generation_only_produces_characters_from_the_vocab():
    """写出来的字必须都在词表里——不然 decode 会直接崩。"""
    text = generate(SMALL_MODEL, TOKENIZER, "床前", 20, temperature=4.0, seed=0)
    assert all(char in TOKENIZER.char_to_id for char in text)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
