"""
这个文件演示本章的产物：把一句话拆成一个个小块，再把小块换成编号。

三步：
1. 先定一份词表——就是我们认识的所有词；
2. 从左往右扫这句话，每一步都尽量切下最长的、词表里有的词；
3. 把切下来的词换成它在词表里的编号。

词表里没有的块（比如"了"、"强"、空格），统一换成一个占位符 <UNK>。
跑起来会看到：每句话都变成了一串编号，程序从"看字符串"变成"看编号"。
"""

# 全书统一使用的词表。顺序是固定的：一个词在这里的下标，就是它的编号
VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]

# 词表里没有的块，统一用这个占位符顶上
UNKNOWN_WORD = "<UNK>"

# 词 -> 编号
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}

# 占位符的编号排在所有词后面
UNKNOWN_ID = len(VOCABULARY)

# 切分时最多往后看几个字：词表里最长的词有多长，就最多看多长
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)

# 全书的 10 句话。前 5 句是科技类，后 5 句是食品类
SENTENCES = [
    "苹果发布新手机",
    "苹果发布新芯片",
    "华为发布新电脑",
    "小米发布新手机",
    "苹果芯片很强",
    "苹果很好吃",
    "苹果很甜",
    "香蕉很好吃",
    "这个苹果真甜",
    "苹果做成派",
]

# 几句语料之外的句子，用来看"拆开"这件事是不是真的通用
NEW_SENTENCES = [
    "苹果发布了新手机",
    "苹果即将发布新手机",
    "苹果配香蕉",
]


def split_sentence(sentence):
    """
    从左往右扫，每一步都尽量切下最长的、词表里有的词。

    切不到词表里的词时，用 <UNK> 顶一个字的位置，然后继续往后扫。
    """
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
            # 词表里没有这一块，占一个位，只往后挪一个字
            words.append(UNKNOWN_WORD)
            position += 1
        else:
            words.append(matched_word)
            position += len(matched_word)
    return words


def to_ids(words):
    """把一串词换成它们的编号。"""
    return [WORD_TO_ID.get(word, UNKNOWN_ID) for word in words]


def main():
    print("=" * 60)
    print("词表：我们认识的词，一共", len(VOCABULARY), "个")
    print("=" * 60)
    for word, index in WORD_TO_ID.items():
        print(f"  编号 {index:>2}  ->  {word}")
    print(f"  编号 {UNKNOWN_ID:>2}  ->  {UNKNOWN_WORD}（词表里没有的词都用它顶上）")
    print()

    print("=" * 60)
    print("把语料里的 10 句话拆开，再换成编号")
    print("=" * 60)
    for sentence in SENTENCES:
        words = split_sentence(sentence)
        ids = to_ids(words)
        print(f"  {sentence}")
        print(f"      拆开: {' | '.join(words)}")
        print(f"      编号: {ids}")
    print()

    print("=" * 60)
    print("换几句语料里没有的句子，试试看")
    print("=" * 60)
    for sentence in NEW_SENTENCES:
        words = split_sentence(sentence)
        ids = to_ids(words)
        print(f"  {sentence}")
        print(f"      拆开: {' | '.join(words)}")
        print(f"      编号: {ids}")
    print()
    print("-" * 60)
    print("从这一行往下，程序看到的就不再是'苹果发布新手机'这根字符串，")
    print("而是一个个编号。编号是程序唯一能拿去做运算的东西。")


if __name__ == "__main__":
    main()
