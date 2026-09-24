"""
第 2 章的测试：测的是本章真正的东西——
一堆手写规则，以及"规则的顺序算不算规则的一部分"。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("after", None)

from after import CORPUS, UNKNOWN_WORD, judge, split_sentence


def judge_with_very_rule_first(words):
    """只把 after.py 里"很"那条规则挪到最前面，其他什么都不改。"""
    if "很" in words:
        return "食品"
    if "芯片" in words:
        return "科技"
    return "不知道"


def test_rules_get_all_corpus_sentences_right():
    for sentence, label in CORPUS:
        assert judge(split_sentence(sentence)) == label


def test_no_corpus_sentence_falls_through_to_unknown():
    """规则是补出来的，但至少在这 10 句上不该漏。"""
    for sentence, _ in CORPUS:
        assert judge(split_sentence(sentence)) != "不知道"


def test_rule_order_changes_the_answer():
    """
    同一句话、同一批规则，只把"很"那条挪到前面，答案就从"科技"变成"食品"。
    """
    words = split_sentence("苹果芯片很强")
    assert words == ["苹果", "芯片", "很", UNKNOWN_WORD]
    assert judge(words) == "科技"                       # after.py 里的顺序
    assert judge_with_very_rule_first(words) == "食品"   # 只换了顺序


def test_apple_sentences_are_decided_by_other_words_not_by_apple():
    """
    "苹果"两边的类别里都有，它自己决定不了任何事。
    这两句能答对，靠的是"甜"和"派"，不是"苹果"。
    """
    assert judge(split_sentence("苹果很甜")) == "食品"
    assert judge(split_sentence("苹果做成派")) == "食品"


def test_rules_cannot_see_words_that_are_not_in_the_vocabulary():
    """
    "苹果好香"里的"香"不在词表里，规则看不见它，
    于是只剩下"苹果"，只能靠兜底去猜——这一句就错了。
    """
    words = split_sentence("苹果好香")
    assert UNKNOWN_WORD in words
    assert judge(words) == "科技"        # 兜底猜的，真实答案是"食品"


def test_rules_only_ever_answer_two_categories():
    answers = {judge(split_sentence(sentence)) for sentence, _ in CORPUS}
    assert answers == {"科技", "食品"}


def test_adding_a_keyword_changes_the_answer():
    """规则只认关键词在不在，不认词序、不认上下文。"""
    assert judge(split_sentence("电脑好贵")) == "科技"
    assert judge(split_sentence("香蕉派很好吃")) == "食品"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
