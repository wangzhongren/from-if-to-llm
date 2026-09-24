"""
这个文件演示"最天真的做法"：不拆句子，把整句话当成一个整体记下来。

计算机拿到"苹果发布新手机"，如果什么都不做，它拿到的就是 7 个字拼成的一根
字符串。它能做的比较只有一种：这根字符串和那根字符串是不是一模一样。

跑起来会看到：原话全都认识，换个写法就全都不认识了。
"""

# 我们"见过"的句子，整条整条地记下来
KNOWN_SENTENCES = [
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

# 意思一样、但写法不一样的句子
REWRITTEN_SENTENCES = [
    "苹果发布了新手机",     # 多了一个"了"
    "苹果发布新手机。",     # 多了一个句号
    "苹果发布新手机！",     # 句号换成感叹号
    "苹果 发布 新手机",     # 中间多了一个空格
    "苹果即将发布新手机",   # 中间多了一个"即将"
]


def has_seen(sentence):
    """这句整话，我们见过吗？"""
    return sentence in KNOWN_SENTENCES


def main():
    print("=" * 60)
    print("第一步：把整句话当成一个整体来记")
    print("=" * 60)
    print()

    print("语料里的 10 句原话：")
    hit = 0
    for sentence in KNOWN_SENTENCES:
        seen = has_seen(sentence)
        hit += 1 if seen else 0
        print(f"  [{'听过' if seen else '没听过'}] {sentence}")
    print()

    print("同样意思、只是换个写法的 5 句话：")
    rewrite_hit = 0
    for sentence in REWRITTEN_SENTENCES:
        seen = has_seen(sentence)
        rewrite_hit += 1 if seen else 0
        print(f"  [{'听过' if seen else '没听过'}] {sentence}")
    print()

    print("-" * 60)
    print(f"原话命中：{hit}/{len(KNOWN_SENTENCES)}")
    print(f"换个写法命中：{rewrite_hit}/{len(REWRITTEN_SENTENCES)}")
    print("-" * 60)
    print()
    print("问题在哪：程序记住的是'这根字符串'，不是'这句话在说什么'。")
    print("多一个'了'、多一个空格，在它眼里就是一根全新的字符串。")


if __name__ == "__main__":
    main()
