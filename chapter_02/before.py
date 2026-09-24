"""
这个文件把第 1 章的成果直接拿来用，然后让它失败。

第 1 章我们把句子变成了编号序列：
    "苹果发布新手机"  ->  [0, 1, 2, 3]

现在我们要回答"这句话是哪一类"。最直接的做法是：
把见过的编号序列连同它的类别一起抄下来，新句子来了就查表。

跑起来会看到：语料里的 10 句全对，语料之外的句子一句也答不上来。
"""

# ---------- 第 1 章的那套东西，原样搬过来 ----------

VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]
UNKNOWN_WORD = "<UNK>"
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
UNKNOWN_ID = len(VOCABULARY)
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)


def split_sentence(sentence):
    words = []
    position = 0
    while position < len(sentence):
        matched_word = None
        for length in range(MAX_WORD_LENGTH, 0, -1):
            piece = sentence[position:position + length]
            if piece in WORD_TO_ID:
                matched_word = piece
                break
        if matched_word is None:
            words.append(UNKNOWN_WORD)
            position += 1
        else:
            words.append(matched_word)
            position += len(matched_word)
    return words


def to_ids(words):
    return [WORD_TO_ID.get(word, UNKNOWN_ID) for word in words]


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


# ---------- 本章的做法：把编号序列和类别一起背下来 ----------

CORPUS = [
    ("苹果发布新手机", "科技"),
    ("苹果发布新芯片", "科技"),
    ("华为发布新电脑", "科技"),
    ("小米发布新手机", "科技"),
    ("苹果芯片很强", "科技"),
    ("苹果很好吃", "食品"),
    ("苹果很甜", "食品"),
    ("香蕉很好吃", "食品"),
    ("这个苹果真甜", "食品"),
    ("苹果做成派", "食品"),
]

# 编号序列 -> 类别。见过的句子，一条一条抄在这里
KNOWN_LABELS = {
    tuple(to_ids(split_sentence(sentence))): label
    for sentence, label in CORPUS
}

# 语料之外的句子，看看它答不答得上来
NEW_SENTENCES = [
    ("苹果发布了新手机", "科技"),
    ("苹果发布新电脑", "科技"),
    ("香蕉很甜", "食品"),
    ("机器人发布新手机", "科技"),
]


def judge_by_lookup(sentence):
    """拿编号序列去查表。表里没有，就只能说不知道。"""
    ids = tuple(to_ids(split_sentence(sentence)))
    return KNOWN_LABELS.get(ids, "不知道")


def main():
    print("=" * 60)
    print("背下来的那张表：")
    print("=" * 60)
    for ids, label in KNOWN_LABELS.items():
        print(f"  {str(list(ids)):<24} -> {label}")
    print()

    print("=" * 60)
    print("语料里的 10 句话")
    print("=" * 60)
    right = 0
    for sentence, label in CORPUS:
        answer = judge_by_lookup(sentence)
        right += 1 if answer == label else 0
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 18)} 答：{answer}")
    print(f"  正确 {right}/{len(CORPUS)}")
    print()

    print("=" * 60)
    print("语料之外的 4 句话")
    print("=" * 60)
    new_right = 0
    for sentence, label in NEW_SENTENCES:
        answer = judge_by_lookup(sentence)
        new_right += 1 if answer == label else 0
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 20)} 答：{pad(answer, 8)} 真实：{label}")
    print(f"  正确 {new_right}/{len(NEW_SENTENCES)}")
    print()
    print("-" * 60)
    print("它能答的只有它背过的那几句。句子里多一个字、少一个字，编号序列")
    print("就对不上，它连'不知道'以外的第二句话都说不出来。")


if __name__ == "__main__":
    main()
