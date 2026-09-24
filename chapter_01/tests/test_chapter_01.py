"""
第 1 章的测试：测的是本章真正的产物——
把句子拆成词、把词换成编号，以及"换个写法还能对上号"这件事。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 别的章节也可能有同名的 after.py，先清掉缓存，确保导入的是本章的
sys.modules.pop("after", None)

from after import (
    SENTENCES,
    UNKNOWN_ID,
    UNKNOWN_WORD,
    VOCABULARY,
    WORD_TO_ID,
    split_sentence,
    to_ids,
)


def test_corpus_sentences_are_split_without_losing_a_character():
    """拆开之后，原来的字一个不多、一个不少（每个 <UNK> 顶一个字）。"""
    for sentence in SENTENCES:
        words = split_sentence(sentence)
        covered = sum(1 if word == UNKNOWN_WORD else len(word) for word in words)
        assert covered == len(sentence)


def test_every_split_word_is_in_the_vocabulary():
    """拆出来的词，要么在词表里，要么是 <UNK>——不能有第三种东西。"""
    for sentence in SENTENCES:
        for word in split_sentence(sentence):
            assert word == UNKNOWN_WORD or word in VOCABULARY


def test_longest_match_picks_whole_words():
    """'这个苹果真甜' 应该拆成 4 个词，而不是 6 个字。"""
    assert split_sentence("这个苹果真甜") == ["这个", "苹果", "真", "甜"]


def test_longest_match_prefers_the_longer_word():
    """'好吃' 是一个词，不能被拆成 '好' 和 '吃'。"""
    assert split_sentence("苹果很好吃") == ["苹果", "很", "好吃"]


def test_unknown_word_gets_the_unknown_id():
    """词表里没有的 '强'，应该拿到 <UNK> 的编号。"""
    words = split_sentence("苹果芯片很强")
    assert words == ["苹果", "芯片", "很", UNKNOWN_WORD]
    assert to_ids(words) == [0, 4, 9, UNKNOWN_ID]


def test_unknown_id_is_not_used_by_any_real_word():
    assert UNKNOWN_ID not in WORD_TO_ID.values()
    # 词表是全书固定的那 16 个词，不许改动
    assert len(VOCABULARY) == 16


def test_same_word_always_gets_the_same_id():
    """'苹果' 在第一句和第十句里，必须是同一个编号。"""
    first = to_ids(split_sentence(SENTENCES[0]))
    last = to_ids(split_sentence(SENTENCES[-1]))
    assert first[0] == last[0] == WORD_TO_ID["苹果"]


def test_all_ids_are_in_range():
    for sentence in SENTENCES:
        for token_id in to_ids(split_sentence(sentence)):
            assert 0 <= token_id <= UNKNOWN_ID


def test_different_wordings_share_ids():
    """换了写法的句子，绝大多数编号仍然对得上——这就是拆开的意义。"""
    original = to_ids(split_sentence("苹果发布新手机"))
    rewritten = to_ids(split_sentence("苹果发布了新手机"))
    shared = [i for i in original if i in rewritten and i != UNKNOWN_ID]
    assert shared == [0, 1, 2, 3]
    assert len(rewritten) == len(original) + 1      # 只多了一个 <UNK>


def test_empty_sentence_splits_to_nothing():
    assert split_sentence("") == []
    assert to_ids([]) == []


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
