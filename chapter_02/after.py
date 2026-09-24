"""
这个文件演示本章的产物：一堆手写的 if/else 规则，把句子分到两个类别里。

规则是从上往下一条条试的，谁先命中就返回谁。
下面这份规则不是一次想出来的，是"跑一遍、看它错在哪、再加一条"补出来的。

请特别留意两件事：
1. 规则的数量，等于"我们见过的错法"的数量；
2. 规则之间的顺序是有意义的，换一下顺序结果就变了。
"""

# ---------- 第 1 章的那套东西，原样搬过来 ----------

VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]
UNKNOWN_WORD = "<UNK>"
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
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


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


# ---------- 本章的产物：一堆规则 ----------

def judge(words):
    """
    给一句拆好的话定一个类别。

    规则从上往下试，谁先命中就返回谁。这是一份补出来的规则，不要随便调整顺序。
    """
    # 补丁 1：看到"手机"，基本就是科技
    if "手机" in words:
        return "科技"

    # 补丁 2：那"芯片"和"电脑"也算
    if "芯片" in words:
        return "科技"
    if "电脑" in words:
        return "科技"

    # 补丁 3：公司名也算科技。华为、小米
    if "华为" in words:
        return "科技"
    if "小米" in words:
        return "科技"

    # 补丁 4："发布"这个词只在科技句里出现过，也算一条
    if "发布" in words:
        return "科技"

    # 补丁 5：食品那边。先说吃的
    if "好吃" in words:
        return "食品"

    # 补丁 6：这条很危险。"很"也出现在"苹果芯片很强"里，
    #         它现在能用，纯粹是因为前面第 2 条已经把它拦下了。
    if "很" in words:
        return "食品"

    # 补丁 7：还有"甜"、"香蕉"、"派"、"做成"
    if "甜" in words:
        return "食品"
    if "香蕉" in words:
        return "食品"
    if "派" in words:
        return "食品"
    if "做成" in words:
        return "食品"

    # 补丁 8：光看见"苹果"怎么办？它两个类别里都有。
    #         想不出办法了，猜一个吧。
    if "苹果" in words:
        return "科技"

    return "不知道"


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


def main():
    print("=" * 60)
    print("手写规则跑语料里的 10 句话")
    print("=" * 60)
    right = 0
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        answer = judge(words)
        right += 1 if answer == label else 0
        mark = "对" if answer == label else "错"
        print(f"  [{mark}] {pad(sentence, 18)} 拆开：{' '.join(words)}")
        print(f"       {pad('', 18)} 答：{answer}")
    print()
    print("-" * 60)
    print(f"正确：{right}/{len(CORPUS)}")
    print("-" * 60)
    print()
    print("这份规则能全对，但它是这么来的：")
    print("  第 1 遍：只写了'手机'一条            ->  2/10")
    print("  第 2 遍：补上芯片、电脑、发布、公司名  ->  5/10")
    print("  第 3 遍：再补上好吃、很、甜……        -> 10/10")
    print()
    print("每一条规则背后，都是我们亲眼看它错了一次。")


if __name__ == "__main__":
    main()
