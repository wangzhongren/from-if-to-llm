"""第 26 章 before.py —— 我们现在的两种分词法，各有各的死法。

第一种：字符级（第 22 章一直用的）。一个字一个 token。
         问题：序列长、语义被切碎。

第二种：词级（第 1 章的办法）。拿一张词表去句子里切词。
         问题一：碰到词表里没有的词，只能吐 <UNK>。
         问题二：想把所有词都收进词表，词表会爆炸。

跑法：

    ./.venv/bin/python chapter_26/before.py
"""


# ---------------------------------------------------------------- 第 1 章的语料

# 第 1 章那批固定的句子，一个字都没改
SENTENCES = {
    "科技": ["苹果发布新手机", "苹果发布新芯片", "华为发布新电脑",
             "小米发布新手机", "苹果芯片很强"],
    "食品": ["苹果很好吃", "苹果很甜", "香蕉很好吃", "这个苹果真甜", "苹果做成派"],
}

# 第 1 章那张固定的词表
VOCABULARY = "苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派".split()

# 几句"以后才出现的"句子：新词、没见过的说法
NEW_SENTENCES = ["苹果不好吃", "华为发布新平板", "小米直播带货", "苹果做成派很好吃"]

UNK = "<UNK>"


# 用的还是书里那批词，只是把它们组合成很多不同的句子（after.py 里是同一份语料）
TECH_SUBJECTS = ["苹果", "华为", "小米"]
TECH_ACTIONS = ["发布新手机", "发布新电脑", "发布新芯片", "芯片很强", "手机很强", "电脑很强"]
FOOD_SUBJECTS = ["苹果", "香蕉", "这个苹果"]
FOOD_ACTIONS = ["很好吃", "很甜", "做成派", "真甜", "真好吃"]
LONG_TEMPLATES = [
    "我昨天在商场看到{subject}刚刚发布的新手机",
    "小王把书给了小李因为他明天考试",
]


def build_corpus(n_sentences=600, seed=0):
    """把书里的词排列组合成一份语料。

    真实语料当然不是这么造出来的。我们只是要一份"片段反复出现、但句子各不相同"的中文。
    """
    import random
    rng = random.Random(seed)
    parts = []
    for index in range(n_sentences):
        if index % 5 == 4:
            template = LONG_TEMPLATES[index % len(LONG_TEMPLATES)]
            parts.append(template.format(subject=rng.choice(TECH_SUBJECTS)) + "。")
        elif index % 2 == 0:
            parts.append(rng.choice(TECH_SUBJECTS) + rng.choice(TECH_ACTIONS) + "。")
        else:
            parts.append(rng.choice(FOOD_SUBJECTS) + rng.choice(FOOD_ACTIONS) + "。")
    return "".join(parts)


# ---------------------------------------------------------------- 第一种：字符级

def char_tokenize(text):
    """一个字符一个 token。第 22 章的模型就是这么喂的。"""
    return list(text)


# ---------------------------------------------------------------- 第二种：词级

def word_tokenize(text, vocabulary=None):
    """拿词表去切句子：从当前位置开始，尽量匹配最长的词。

    这就是第 1 章"先把一句话拆开"的做法，也是中文分词最朴素的一种。
    """
    vocabulary = vocabulary or VOCABULARY
    longest = max(len(w) for w in vocabulary)
    tokens = []
    position = 0
    while position < len(text):
        for length in range(min(longest, len(text) - position), 0, -1):
            piece = text[position:position + length]
            if piece in vocabulary:
                tokens.append(piece)
                position += length
                break
        else:
            # 一个词都没匹配上：这个字我们不认识
            tokens.append(UNK)
            position += 1
    return tokens


def count_ngrams(text, n):
    """语料里有多少个不同的 n 字片段。"""
    return len({text[i:i + n] for i in range(len(text) - n + 1)})


def readable(number):
    """把大数字写成人话：3,000 / 900 万 / 270 亿 / 81 万亿。"""
    if number >= 1e12:
        return f"{number / 1e12:,.0f} 万亿"
    if number >= 1e8:
        return f"{number / 1e8:,.0f} 亿"
    if number >= 1e4:
        return f"{number / 1e4:,.0f} 万"
    return f"{number:,}"


def main():
    corpus = build_corpus()

    print("=" * 64)
    print("第一种：字符级")
    print("=" * 64)
    for sentence in ["苹果很好吃", "苹果不好吃", "床前明月光"]:
        tokens = char_tokenize(sentence)
        print(f"{sentence} -> {tokens}   （{len(tokens)} 个 token）")
    print()
    print("每个字都有自己的 token，所以：")
    print(f"  语料里不同的字：{count_ngrams(corpus, 1)} 个")
    print(f"  一句话有几个字，就有几个 token")
    print()

    print("=" * 64)
    print("第二种：词级（拿第 1 章的词表去切）")
    print("=" * 64)
    print(f"词表大小：{len(VOCABULARY)} —— {VOCABULARY}")
    print()
    for sentence in [n for group in SENTENCES.values() for n in group] + NEW_SENTENCES:
        tokens = word_tokenize(sentence)
        unknown = tokens.count(UNK)
        mark = "  <- 有生词！" if unknown else ""
        print(f"{sentence} -> {tokens}{mark}")
    print()

    print("=" * 64)
    print("词级分词的两个麻烦")
    print("=" * 64)
    print("麻烦一：生词。一个'不'字就能让整句话的意思反过来，而它不在词表里。")
    print("（上面第一个吐生词的句子'苹果芯片很强'，还是第 1 章自己的句子：")
    print("  词表里有'很'有'甜'，就是没有'强'。词表是人手写的，写不全。）")
    print()
    print("麻烦二：如果想把词都收进去，词表要多大？")
    print(f"  我们的语料一共 {len(corpus)} 个字。")
    for n in (1, 2, 3, 4, 5):
        print(f"  语料里不同的 {n} 字片段：{count_ngrams(corpus, n):,} 个")
    print()
    print("  语料越大，新片段就越多，词表跟着一直长。理论上的上限更吓人：")
    for n in (1, 2, 3, 4):
        print(f"  常用汉字 3000 个，{n} 字组合最多有 {readable(3000 ** n)}个")
    print()
    print("第 2 章那个'几万个 if'的问题，换了一件衣服就回来了。")


if __name__ == "__main__":
    main()
