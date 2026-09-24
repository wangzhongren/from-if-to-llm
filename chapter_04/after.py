"""
这个文件演示本章的产物：让程序自己修改那些数字。

做法只有一条规则：
    这句话答错了，就把正确答案那一类的数字往大调一点，
    把答错那一类的数字往小调一点。
答对了就什么都不做。

我们不填任何数字——所有权重一开始全是 0，全靠程序自己改。
这个"根据对错去改数字"的过程，以后我们叫它训练。

跑起来会看到：错句数一轮轮掉下去，最后 10 句全对。
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

# 两个类别固定按这个顺序排：0 号是科技，1 号是食品
CATEGORIES = ["科技", "食品"]

# 每次改多少。这一章先固定成 1.0——第 5 章我们要回头算这笔账。
STEP = 1.0


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


# ---------- 本章的产物：一张会被改来改去的表 ----------

# 每个词后面跟着两个权重：[给"科技"的权重, 给"食品"的权重]。
# 一开始全是 0——我们不填任何数字。
WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}

# 有没有的词（<UNK>）没资格进这张表：我们连它是什么字都不知道，
# 更不知道该给它多少权重。


def score(words):
    """把每个词的两个权重分别加起来，得到两个分数。"""
    scores = [0.0, 0.0]
    for word in words:
        if word not in WEIGHT_TABLE:
            continue
        weights = WEIGHT_TABLE[word]
        scores[0] += weights[0]
        scores[1] += weights[1]
    return scores


def predict(words):
    """分数高的赢。两个分数一样高的时候，先算科技。"""
    scores = score(words)
    return CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]


def adjust(words, correct_category, wrong_category, step):
    """
    改权重：这句话里的每个词，都往正确的那边挪一点、从错误的那边退一点。

    注意这里是按"出现次数"改的：一个词在句子里出现两次，就改两次。
    """
    correct_index = CATEGORIES.index(correct_category)
    wrong_index = CATEGORIES.index(wrong_category)
    for word in words:
        if word not in WEIGHT_TABLE:
            continue
        WEIGHT_TABLE[word][correct_index] += step
        WEIGHT_TABLE[word][wrong_index] -= step


def train_one_round(corpus, step):
    """把语料从头到尾走一遍。每答错一句，就改一次权重。"""
    wrong_count = 0
    for sentence, label in corpus:
        words = split_sentence(sentence)
        answer = predict(words)
        if answer == label:
            continue
        wrong_count += 1
        adjust(words, correct_category=label, wrong_category=answer, step=step)
    return wrong_count


def measure(corpus):
    right = 0
    for sentence, label in corpus:
        if predict(split_sentence(sentence)) == label:
            right += 1
    return right


def main():
    print("=" * 66)
    print("起点：所有词的权重都是 0")
    print("=" * 66)
    print(f"  正确：{measure(CORPUS)}/10")
    print("  解释：全 0 的时候两个分数都是 0，按规矩先算科技——")
    print("        5 句科技正好被蒙对了，5 句食品全错。")
    print()

    print("=" * 66)
    print("训练：错了就改，对了不动")
    print("=" * 66)
    print("  轮次   答错    正确")
    for round_number in range(1, 11):
        wrong_count = train_one_round(CORPUS, step=STEP)
        right = measure(CORPUS)
        bar = "#" * wrong_count
        print(f"   {round_number:>2}    {wrong_count:>2}     {right:>2}/10  {bar}")
        if wrong_count == 0:
            break
    print()

    print("=" * 66)
    print("训练完的权重表")
    print("=" * 66)
    for word, weights in WEIGHT_TABLE.items():
        print(f"  {pad(word, 8)} 科技 {weights[0]:>5.1f}   食品 {weights[1]:>5.1f}")
    print()

    print("=" * 66)
    print("检查每一句")
    print("=" * 66)
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        scores = score(words)
        answer = predict(words)
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 18)} 科技 {scores[0]:>5.1f} / 食品 {scores[1]:>5.1f}")
    print()
    print("-" * 66)
    print(f"正确：{measure(CORPUS)}/10")
    print()
    print("这 32 个数字，我们一个都没填。我们只写了一条规则：")
    print("    答错了，就把正确的往大调一点、错误的往小调一点。")
    print()
    print("但那条规则里有个数是我们定的：每次调 1.0。")
    print("为什么是 1.0？0.1 行不行？100 行不行？我们并不知道。")


if __name__ == "__main__":
    main()
