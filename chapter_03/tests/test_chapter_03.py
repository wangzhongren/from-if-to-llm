"""
第 3 章的测试：测的是本章真正的东西——
一句话怎么变成两个分数，以及"一个词能不能同时给两边分量"。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("after", None)

from after import (
    COEFFICIENT_TABLE,
    CORPUS,
    UNKNOWN_WORD,
    VOCABULARY,
    judge,
    score,
    split_sentence,
)


def test_weighted_scores_get_all_corpus_sentences_right():
    for sentence, label in CORPUS:
        assert judge(split_sentence(sentence)) == label


def test_score_is_the_sum_of_the_coefficients():
    """「苹果发布新手机」= 苹果 + 发布 + 新 + 手机，一个词都不多、不少。"""
    technology_score, food_score = score(split_sentence("苹果发布新手机"))
    expected_technology = (COEFFICIENT_TABLE["苹果"][0] + COEFFICIENT_TABLE["发布"][0]
                           + COEFFICIENT_TABLE["新"][0] + COEFFICIENT_TABLE["手机"][0])
    expected_food = (COEFFICIENT_TABLE["苹果"][1] + COEFFICIENT_TABLE["发布"][1]
                     + COEFFICIENT_TABLE["新"][1] + COEFFICIENT_TABLE["手机"][1])
    assert abs(technology_score - expected_technology) < 1e-9
    assert abs(food_score - expected_food) < 1e-9
    assert abs(technology_score - 2.7) < 1e-9
    assert abs(food_score - 0.9) < 1e-9


def test_one_word_can_feed_both_sides():
    """
    这一章和上一章最大的不同：'苹果'不用选边站了。
    它同时给两个类别分量，只是大小不同。
    """
    technology_coefficient, food_coefficient = COEFFICIENT_TABLE["苹果"]
    assert technology_coefficient > 0
    assert food_coefficient > 0
    assert food_coefficient != technology_coefficient


def test_every_vocabulary_word_has_a_row():
    """词表里的每个词都得在系数表里有位置，不能漏。"""
    for word in VOCABULARY:
        assert word in COEFFICIENT_TABLE


def test_no_row_is_all_zeros():
    """一行两个系数都是 0，等于这个词没被用上——基本可以确定是漏掉了。"""
    for word, (technology_coefficient, food_coefficient) in COEFFICIENT_TABLE.items():
        assert technology_coefficient != 0 or food_coefficient != 0, word


def test_a_word_that_appears_twice_counts_twice():
    assert score(["苹果", "苹果"]) == (0.4, 1.8)


def test_unknown_words_do_not_change_the_score():
    assert score(["苹果", UNKNOWN_WORD]) == score(["苹果"])


def test_score_is_what_decides_not_a_rule():
    """
    「苹果芯片很强」这句是本章的关键：它在数票法里被判成食品，
    在这里靠 1.7 : 1.2 赢下来——赢在分数，不是赢在某条规则。
    """
    technology_score, food_score = score(split_sentence("苹果芯片很强"))
    assert technology_score > food_score
    assert technology_score == 1.7
    assert food_score == 1.2


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
