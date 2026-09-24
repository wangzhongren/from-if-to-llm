"""
这个文件把第 4 章的代码原样搬过来，然后问一句：
"它怎么知道自己是好是坏？"

第 4 章的训练规则是"答错了就改权重"。它靠什么知道训练该停了？
靠"这一轮答错了几句"。这是一个整数，取值只有 0 到 10。

跑起来会看到：第 4 轮之后，这个数一直是 0。
它说"没有错了"——然后就没话说了。
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

CATEGORIES = ["科技", "食品"]
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


WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}


def score(words):
    scores = [0.0, 0.0]
    for word in words:
        if word not in WEIGHT_TABLE:
            continue
        weights = WEIGHT_TABLE[word]
        scores[0] += weights[0]
        scores[1] += weights[1]
    return scores


def predict(words):
    scores = score(words)
    return CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]


def train_one_round(corpus, step):
    """第 4 章的那条规则：答错了就改，对了不动。"""
    wrong_count = 0
    for sentence, label in corpus:
        words = split_sentence(sentence)
        answer = predict(words)
        if answer == label:
            continue
        wrong_count += 1
        correct_index = CATEGORIES.index(label)
        wrong_index = CATEGORIES.index(answer)
        for word in words:
            if word not in WEIGHT_TABLE:
                continue
            WEIGHT_TABLE[word][correct_index] += step
            WEIGHT_TABLE[word][wrong_index] -= step
    return wrong_count


def count_wrong(corpus):
    """第 4 章唯一的尺子：这一轮答错了几句。"""
    wrong = 0
    for sentence, label in corpus:
        if predict(split_sentence(sentence)) != label:
            wrong += 1
    return wrong


def main():
    print("=" * 66)
    print("先用第 4 章的规则训练，看它的尺子怎么读数")
    print("=" * 66)
    print("  轮次   这把尺子的读数（答错了几句）")
    for round_number in range(1, 11):
        wrong_count = train_one_round(CORPUS, step=STEP)
        print(f"   {round_number:>2}     {wrong_count:>2}")
    print()
    print("  第 4 轮之后，读数一直是 0。")
    print("  这把尺子的取值范围是 0 到 10——一共 11 个刻度。")
    print()

    print("=" * 66)
    print("训练完的权重表")
    print("=" * 66)
    for word, weights in WEIGHT_TABLE.items():
        print(f"  {pad(word, 8)} 科技 {weights[0]:>5.1f}   食品 {weights[1]:>5.1f}")
    print()

    print("=" * 66)
    print("它答对了，但它是'笃定'还是'勉强'？")
    print("=" * 66)
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        scores = score(words)
        answer = predict(words)
        margin = abs(scores[0] - scores[1])
        print(f"  {pad(sentence, 18)} 科技 {scores[0]:>5.1f} / 食品 {scores[1]:>5.1f}"
              f"   赢 {margin:>4.1f}")
    print()
    print("-" * 66)
    print("看这两行：'苹果芯片很强'只赢了 2.0 分，'这个苹果真甜'赢了 8.0 分。")
    print()
    print("可第 4 章的尺子对这两句的读数是一模一样的：都对。")
    print("尺子读到 0 了，它就到头了——我们没法从它那里知道：")
    print("    哪一句其实快输了？")
    print("    这个表和另一个同样 0 错的表，哪个更好？")
    print("    训练还能不能继续变好？")


if __name__ == "__main__":
    main()
