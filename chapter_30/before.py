"""第 30 章 before.py —— 回到第一行代码。

这是全书的第一段代码（第 2 章）。它做的事情是：拿一堆 if/else，给句子分类。

    if "手机" in words:
        print("科技")

那时候我们问：几万个词，难道要写几万个 if？
这个文件就是那堆 if。它会在这一章里，和 29 章之后的模型站在同一个测试上。

跑法：

    ./.venv/bin/python chapter_30/before.py
"""


# ---------------------------------------------------------------- 第 1 章的语料

SENTENCES = {
    "科技": ["苹果发布新手机", "苹果发布新芯片", "华为发布新电脑",
             "小米发布新手机", "苹果芯片很强"],
    "食品": ["苹果很好吃", "苹果很甜", "香蕉很好吃", "这个苹果真甜", "苹果做成派"],
}

# 第 1 章那张固定的词表：先用它把句子切成词
VOCABULARY = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

# 这一章用的测试集：前面十句是第 1 章的原句，后面是"以后才出现"的句子
TEST_SET = [
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
    ("小米直播带货", "科技"),
    ("华为发布新平板", "科技"),
    ("苹果不好吃", "食品"),
    ("香蕉做成派很好吃", "食品"),
    ("苹果手机很甜", "说不清"),
    ("这个手机真好吃", "说不清"),
]


def split_words(sentence):
    """第 1 章的分词：拿词表去句子里切。切不出来的字先丢在一边。"""
    longest = max(len(word) for word in VOCABULARY)
    words = []
    position = 0
    while position < len(sentence):
        for length in range(min(longest, len(sentence) - position), 0, -1):
            piece = sentence[position:position + length]
            if piece in VOCABULARY:
                words.append(piece)
                position += length
                break
        else:
            position += 1          # 不认识的词，跳过
    return words


# ---------------------------------------------------------------- 第一行代码

TECH_WORDS = ["手机", "芯片", "电脑", "发布", "华为", "小米"]
FOOD_WORDS = ["好吃", "很甜", "甜", "香蕉"]


def classify(words):
    """第 2 章那个 if/else 分类器。规则的顺序就是它的全部逻辑。"""
    for word in words:
        if word in TECH_WORDS:
            return "科技"
        if word in FOOD_WORDS:
            return "食品"
    return "食品"      # 一个关键词都没碰上，先猜食品


def main():
    print("=" * 70)
    print("第一行代码")
    print("=" * 70)
    print('    if "手机" in words:')
    print('        print("科技")')
    print()
    print("它后面跟着一长串 if/elif，这里完整地重写一遍（第 2 章的样子）。")
    print()

    print("=" * 70)
    print("它怎么分类")
    print("=" * 70)
    print(f"{'句子':<18}{'切出来的词':<34}{'它说'}")
    print("-" * 70)
    correct = 0
    total = 0
    for sentence, label in TEST_SET:
        words = split_words(sentence)
        guess = classify(words)
        mark = ""
        if label != "说不清":
            total += 1
            correct += guess == label
            mark = "" if guess == label else "  <- 错了"
        print(f"{sentence:<18}{' '.join(words):<34}{guess}{mark}")
    print("-" * 70)
    print(f"在 {total} 句能分出对错的句子上，它对了 {correct} 句。")
    print()
    print("注意几件事：")
    print("  1. 14 句全对。这批句子它当然全对 —— 因为规则就是照着它们写的。")
    print("  2. '小米直播带货'它答对了，但靠的是'小米'这两个字在科技词表里，")
    print("     不是因为看懂了'直播带货'。换一个没写进词表的品牌，它就没辙了。")
    print("  3. '苹果手机很甜'它判成科技 —— 因为'手机'排在'甜'前面。")
    print("     规则有顺序，而这个顺序是人随手写的。")
    print("  4. 每一句话为什么这么判，我们都能指着某一行 if 说清楚。")
    print("     这是它的优点，也是它的天花板：**只有我们想到的规律，它才会。**")
    print()
    print("问题来了：我们花了 29 章造出来的那个模型，")
    print("在这同一批句子上，能做得更好吗？")


if __name__ == "__main__":
    main()
