"""
第 4 章的测试：测的是本章真正的东西——
"答错了就改权重"这一条规则，能不能把一张全 0 的表训成一张全对的表。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("after", None)

from after import (
    CORPUS,
    STEP,
    UNKNOWN_WORD,
    VOCABULARY,
    WEIGHT_TABLE,
    adjust,
    measure,
    predict,
    split_sentence,
    train_one_round,
)


@pytest.fixture(autouse=True)
def restore_weight_table():
    """权重表是全局的，每个测试跑完都恢复原样，免得互相影响。"""
    saved = {word: list(weights) for word, weights in WEIGHT_TABLE.items()}
    yield
    for word, weights in saved.items():
        WEIGHT_TABLE[word][:] = weights


def train_until_converged(max_rounds=20):
    for _ in range(max_rounds):
        if train_one_round(CORPUS, step=STEP) == 0:
            break


def test_weights_start_at_zero():
    """我们不填任何数字——所有词的权重一开始都是 0。"""
    for weights in WEIGHT_TABLE.values():
        assert weights == [0.0, 0.0]


def test_training_gets_all_corpus_sentences_right():
    train_until_converged()
    assert measure(CORPUS) == 10


def test_a_mistake_only_touches_the_words_in_that_sentence():
    """改权重的时候，只有这句话里出现过的词被改。"""
    adjust(split_sentence("苹果很甜"), correct_category="食品", wrong_category="科技", step=1.0)
    assert WEIGHT_TABLE["苹果"] == [-1.0, 1.0]
    assert WEIGHT_TABLE["甜"] == [-1.0, 1.0]
    assert WEIGHT_TABLE["芯片"] == [0.0, 0.0]      # 这句里没有"芯片"
    assert WEIGHT_TABLE["香蕉"] == [0.0, 0.0]


def test_adjust_pushes_the_correct_category_up_and_the_wrong_one_down():
    adjust(["芯片"], correct_category="科技", wrong_category="食品", step=1.0)
    assert WEIGHT_TABLE["芯片"] == [1.0, -1.0]


def test_a_word_that_appears_twice_is_adjusted_twice():
    adjust(["苹果", "苹果"], correct_category="食品", wrong_category="科技", step=1.0)
    assert WEIGHT_TABLE["苹果"] == [-2.0, 2.0]


def test_unknown_words_never_get_weights():
    """<UNK> 连是什么字都不知道，不该有权重。"""
    assert UNKNOWN_WORD not in WEIGHT_TABLE
    assert set(WEIGHT_TABLE.keys()) == set(VOCABULARY)
    adjust(split_sentence("苹果芯片很强"), correct_category="科技", wrong_category="食品", step=1.0)
    assert UNKNOWN_WORD not in WEIGHT_TABLE


def test_a_wrong_sentence_triggers_an_update():
    """权重全 0 时，「苹果很甜」会被答成科技，所以它该动手改。"""
    train_one_round([("苹果很甜", "食品")], step=STEP)
    assert WEIGHT_TABLE["苹果"] == [-1.0, 1.0]


def test_a_correct_sentence_does_not_change_anything():
    """答对了就什么都不做——这是这条规则的另一半。"""
    train_one_round([("苹果很甜", "食品")], step=STEP)          # 先让它改一次
    trained = {word: list(weights) for word, weights in WEIGHT_TABLE.items()}
    train_one_round([("苹果很甜", "食品")], step=STEP)          # 这句话已经答对了
    assert WEIGHT_TABLE == trained


def test_training_is_reproducible():
    """同样的起点、同样的语料，跑两遍必须一模一样。"""
    train_until_converged()
    first = {word: list(weights) for word, weights in WEIGHT_TABLE.items()}

    for weights in WEIGHT_TABLE.values():
        weights[:] = [0.0, 0.0]
    train_until_converged()
    second = {word: list(weights) for word, weights in WEIGHT_TABLE.items()}

    assert first == second


def test_extra_training_does_not_break_a_converged_table():
    train_until_converged()
    for _ in range(5):
        train_one_round(CORPUS, step=STEP)
    assert measure(CORPUS) == 10


def test_a_tie_between_two_scores_falls_back_to_the_first_category():
    """两个分数都是 0 的时候，按规矩先算科技——所以全 0 的表也能'蒙对'5 句。"""
    assert predict(split_sentence("苹果发布新手机")) == "科技"
    assert measure(CORPUS) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
