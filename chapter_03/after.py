"""
这个文件演示本章的产物：把 if 变成数字。

第 2 章的规则是"句子里有没有这个词"，只有"有"和"没有"两种可能。
这一章我们给每个词两个系数：
    这个词每出现一次，给"科技"加多少分、给"食品"加多少分。

句子不再"命中一条规则"，而是攒出两个分数，谁高算谁。

跑起来会看到：10 句全对，而且每一句都看得出来"赢了多少"。
"""

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


# ---------- 本章的产物：一张写满数字的表 ----------

# 每个词后面跟着两个数：(给"科技"加多少分, 给"食品"加多少分)。
# 这两个数是我们自己填的——先记住这一点，第 4 章要拿它开刀。
COEFFICIENT_TABLE = {
    "苹果": (0.2, 0.9),     # 两边都喂一点，但食品那边多一点
    "发布": (0.8, 0.0),
    "新":   (0.5, 0.0),
    "手机": (1.2, 0.0),
    "芯片": (1.5, 0.0),
    "电脑": (1.0, 0.0),
    "华为": (1.0, 0.0),
    "小米": (1.0, 0.0),
    "好吃": (0.0, 1.5),
    "很":   (0.0, 0.3),     # 也很小：它说不准，只是稍微倾向食品
    "甜":   (0.0, 0.8),
    "香蕉": (0.0, 1.2),
    "这个": (0.0, 0.2),
    "真":   (0.0, 0.6),
    "做成": (0.0, 0.5),
    "派":   (0.0, 0.5),
}


def score(words):
    """
    给一句话算两个分数。

    每个词出现一次，就把它那两个系数分别加到两个分数上。
    句子里认不出来的词（<UNK>）在表里没有，跳过。
    """
    technology_score = 0.0
    food_score = 0.0
    for word in words:
        if word not in COEFFICIENT_TABLE:
            continue
        technology_coefficient, food_coefficient = COEFFICIENT_TABLE[word]
        technology_score += technology_coefficient
        food_score += food_coefficient
    return technology_score, food_score


def judge(words):
    """谁的分数高算谁。分数一样高的时候，先算科技。"""
    technology_score, food_score = score(words)
    return "科技" if technology_score >= food_score else "食品"


def main():
    print("=" * 66)
    print("系数表：每个词给两边的分数")
    print("=" * 66)
    for word, (technology_coefficient, food_coefficient) in COEFFICIENT_TABLE.items():
        print(f"  {pad(word, 8)} 科技 +{technology_coefficient}   食品 +{food_coefficient}")
    print()

    print("=" * 66)
    print("打分的结果")
    print("=" * 66)
    right = 0
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        technology_score, food_score = score(words)
        answer = judge(words)
        right += 1 if answer == label else 0
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 18)} 科技 {technology_score:>4.1f} / 食品 {food_score:>4.1f}"
              f"   答：{pad(answer, 6)} 真实：{label}")
    print()
    print("-" * 66)
    print(f"正确：{right}/{len(CORPUS)}")
    print("-" * 66)
    print()
    print("对比一下这两句：")
    print("  苹果发布新手机   科技 2.7 / 食品 0.9   赢 1.8 分——很稳")
    print("  苹果芯片很强     科技 1.7 / 食品 1.2   只赢 0.5 分——有点勉强")
    print()
    print("分数不只是'对/错'，它还带着'有多确定'。")


if __name__ == "__main__":
    main()
