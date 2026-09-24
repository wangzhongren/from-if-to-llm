"""
这个文件把第 3 章的做法摆到台面上，然后暴露它的软肋。

第 3 章我们填出了一张系数表，10 句全对。但那 32 个数字是人填的——
我们反复跑、反复改，改到全对为止。

这个文件问一个问题：如果换一组数字呢？

跑起来会看到：手填的那张全对，随机填的那张差得远。
而这两张表，程序都只当成"一堆数字"来看——它没有任何办法把差的那张变好。
"""

import random

VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]
UNKNOWN_WORD = "<UNK>"
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)

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


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


# ---------- 第 3 章手填的那张表 ----------

HAND_FILLED_TABLE = {
    "苹果": (0.2, 0.9),
    "发布": (0.8, 0.0),
    "新":   (0.5, 0.0),
    "手机": (1.2, 0.0),
    "芯片": (1.5, 0.0),
    "电脑": (1.0, 0.0),
    "华为": (1.0, 0.0),
    "小米": (1.0, 0.0),
    "好吃": (0.0, 1.5),
    "很":   (0.0, 0.3),
    "甜":   (0.0, 0.8),
    "香蕉": (0.0, 1.2),
    "这个": (0.0, 0.2),
    "真":   (0.0, 0.6),
    "做成": (0.0, 0.5),
    "派":   (0.0, 0.5),
}


def make_random_table(seed):
    """随便填一张表：每个词的系数都是 0 到 2 之间的随机数。"""
    random.seed(seed)
    return {word: (round(random.uniform(0.0, 2.0), 2),
                   round(random.uniform(0.0, 2.0), 2))
            for word in VOCABULARY}


# ---------- 打分和判断，和第 3 章一样 ----------

def score(words, table):
    technology_score = 0.0
    food_score = 0.0
    for word in words:
        if word not in table:
            continue
        technology_coefficient, food_coefficient = table[word]
        technology_score += technology_coefficient
        food_score += food_coefficient
    return technology_score, food_score


def judge(words, table):
    technology_score, food_score = score(words, table)
    return "科技" if technology_score >= food_score else "食品"


def measure(table):
    right = 0
    for sentence, label in CORPUS:
        if judge(split_sentence(sentence), table) == label:
            right += 1
    return right


def main():
    print("=" * 66)
    print("表 A：第 3 章我们手填的那张")
    print("=" * 66)
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        technology_score, food_score = score(words, HAND_FILLED_TABLE)
        print(f"  {pad(sentence, 18)} 科技 {technology_score:>4.1f} / 食品 {food_score:>4.1f}")
    print(f"  正确：{measure(HAND_FILLED_TABLE)}/10")
    print()

    print("=" * 66)
    print("表 B：随便填一张（同样的形状，数字是随机的）")
    print("=" * 66)
    random_table = make_random_table(seed=42)
    for word in VOCABULARY:
        technology_coefficient, food_coefficient = random_table[word]
        print(f"  {pad(word, 8)} 科技 {technology_coefficient:>4.2f} / 食品 {food_coefficient:>4.2f}")
    print()
    print(f"  正确：{measure(random_table)}/10")
    print()

    print("=" * 66)
    print("两张表放在一起看")
    print("=" * 66)
    print(f"  表 A（人填的）    正确 {measure(HAND_FILLED_TABLE)}/10")
    print(f"  表 B（随便填的）  正确 {measure(random_table)}/10")
    print()
    print("-" * 66)
    print("表 A 是怎么来的？我们跑一遍、看哪句错了、改一个数字、再跑一遍……")
    print("改到 10/10 就停手。也就是说：这 32 个数字是我们'看着答案'调出来的。")
    print()
    print("程序在这件事里干了什么？它只负责算分数和比大小。")
    print("哪个数字该改成多少，它一点想法都没有。")


if __name__ == "__main__":
    main()
