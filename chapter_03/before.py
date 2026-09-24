"""
这个文件把第 2 章的写法机械化，然后让它失败。

第 2 章我们手写了 8 条规则。现在换个思路：
规则不用手写，让程序自己数——看每个词在语料里"更常出现在哪一类"，
然后把它归到那一类去。

新句子来了，就数一数它里面的词各属于哪一类，票多的赢。

跑起来会看到：10 句里错了 1 句。错的那一句很值得看。
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


# ---------- 本章的做法：每个词投一票 ----------

def count_appearances(corpus):
    """数一数每个词在两类句子里各出现过几次。"""
    technology_count = {word: 0 for word in VOCABULARY}
    food_count = {word: 0 for word in VOCABULARY}
    for sentence, label in corpus:
        for word in split_sentence(sentence):
            if word not in WORD_TO_ID:
                continue
            if label == "科技":
                technology_count[word] += 1
            else:
                food_count[word] += 1
    return technology_count, food_count


def build_word_to_category(corpus):
    """按多数派，把每个词归到一个类别里。一个词只能选一边。"""
    technology_count, food_count = count_appearances(corpus)
    table = {}
    for word in VOCABULARY:
        if technology_count[word] > food_count[word]:
            table[word] = "科技"
        else:
            table[word] = "食品"
    return table


WORD_TO_CATEGORY = build_word_to_category(CORPUS)


def count_votes(words):
    """数票：这句话里的词，各给两边投了几票。"""
    technology_votes = 0
    food_votes = 0
    for word in words:
        category = WORD_TO_CATEGORY.get(word)
        if category == "科技":
            technology_votes += 1
        elif category == "食品":
            food_votes += 1
    return technology_votes, food_votes


def judge(words):
    technology_votes, food_votes = count_votes(words)
    if technology_votes == food_votes:
        return "平局"
    return "科技" if technology_votes > food_votes else "食品"


def main():
    print("=" * 66)
    print("程序自己数出来的表：每个词归哪一类（按多数派）")
    print("=" * 66)
    technology_count, food_count = count_appearances(CORPUS)
    for word in VOCABULARY:
        print(f"  {pad(word, 8)} 科技 {technology_count[word]} 句 / 食品 {food_count[word]} 句"
              f"   ->  {WORD_TO_CATEGORY[word]}")
    print()

    print("=" * 66)
    print("数票的结果")
    print("=" * 66)
    right = 0
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        technology_votes, food_votes = count_votes(words)
        answer = judge(words)
        right += 1 if answer == label else 0
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 18)} 科技 {technology_votes} 票 / 食品 {food_votes} 票"
              f"  答：{pad(answer, 6)} 真实：{label}")
    print()
    print("-" * 66)
    print(f"正确：{right}/{len(CORPUS)}")
    print("-" * 66)
    print()
    print("错的是「苹果芯片很强」。看它的票：")
    print("  '苹果' 被程序归到了食品（它在食品句里出现 4 次，在科技句里 3 次）；")
    print("  '很'   也被归到了食品（食品 3 次，科技 1 次）；")
    print("  '芯片' 归了科技。")
    print()
    print("于是这句里有 2 张食品票、1 张科技票，答成了食品。")
    print("可是'苹果'和'很'两个词，本来就是两边都有的。")


if __name__ == "__main__":
    main()
