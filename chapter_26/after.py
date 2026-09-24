"""第 26 章 after.py —— 一个能跑的 mini-BPE。

BPE（Byte Pair Encoding）的想法只有一句话：

    从"每个字符一个 token"开始，反复找出**当前最常挨在一起的两个 token**，
    把它们合并成一个新 token。

合并多少次，词表就长多大。词表大小由我们定，而不是由语料里"有哪些词"定。

这份代码有两个入口：

    字符版：token 是一个个汉字            train_bpe(corpus, 200)
    字节版：token 是一个个字节（UTF-8）    train_byte_bpe(corpus, 200)

**它们是同一个算法**，只是"最小的单位"不同。区别在哪里，跑一遍就看出来了。

跑法：

    ./.venv/bin/python chapter_26/after.py
"""

import random

# ---------------------------------------------------------------- 语料

# 用的还是书里那批词，只是把它们组合成很多不同的句子。
# 这样"苹果""发布""手机"这些片段会反复出现，但没有哪一句被整句抄下来。
TECH_SUBJECTS = ["苹果", "华为", "小米"]
TECH_ACTIONS = ["发布新手机", "发布新电脑", "发布新芯片", "芯片很强", "手机很强", "电脑很强"]
FOOD_SUBJECTS = ["苹果", "香蕉", "这个苹果"]
FOOD_ACTIONS = ["很好吃", "很甜", "做成派", "真甜", "真好吃"]

# 第 13 章那两句长句，也放进语料里（品牌名换着来）
LONG_TEMPLATES = [
    "我昨天在商场看到{subject}刚刚发布的新手机",
    "小王把书给了小李因为他明天考试",
]


def build_corpus(n_sentences=600, seed=0):
    """把书里的词排列组合成一份语料。

    真实语料当然不是这么造出来的。我们只是要一份"片段反复出现、但句子各不相同"的中文，
    让 BPE 有东西可学。第 28 章还会用同一套办法造更大的语料。
    """
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


# ---------------------------------------------------------------- BPE 本体
#
# 三个函数，就是 BPE 的全部。注意它们不关心 token 到底是什么：
# 汉字也好、字节也好，只要能"拼起来"就行。

def count_pairs(tokens):
    """数一数：相邻的两个 token 一起出现了多少次。"""
    pairs = {}
    for left, right in zip(tokens, tokens[1:]):
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
    return pairs


def merge_pair(tokens, pair):
    """把序列里所有的 `pair` 合并成一个 token：A B -> AB。"""
    merged = []
    position = 0
    while position < len(tokens):
        if (position < len(tokens) - 1
                and tokens[position] == pair[0] and tokens[position + 1] == pair[1]):
            merged.append(pair[0] + pair[1])
            position += 2
        else:
            merged.append(tokens[position])
            position += 1
    return merged


def train_bpe(text, num_merges, unit=None):
    """训练：反复合并当前最高频的相邻对，合并 num_merges 次。

    返回按顺序排好的合并列表 —— 这个顺序就是"词表是怎么长出来的"的过程。
    """
    tokens = unit(text) if unit else list(text)
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(tokens)
        if not pairs:
            break
        # 频率最高的那个；频率相同时取字典序大的，纯粹为了结果稳定（种子之外再稳一层）
        best = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best)
        tokens = merge_pair(tokens, best)
    return merges


def encode(text, merges, unit=None):
    """编码：把合并规则按学到的顺序，一条一条用上去。"""
    tokens = unit(text) if unit else list(text)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens


def decode(tokens):
    """解码：把 token 拼回去。因为每次合并只是把两个 token 粘在一起，
    所以只要按顺序拼起来就能还原原文。"""
    return "".join(tokens)


def token_count(text, merges, unit=None):
    return len(encode(text, merges, unit=unit))


# ---------------------------------------------------------------- 字节版

def to_bytes(text):
    """把一句话拆成一串字节。中文的每个字在 UTF-8 里占 3 个字节。"""
    return [bytes([b]) for b in text.encode("utf-8")]


def from_bytes(tokens):
    return b"".join(tokens).decode("utf-8")


# ---------------------------------------------------------------- 主程序

def show(title):
    print("=" * 64)
    print(title)
    print("=" * 64)


def main():
    corpus = build_corpus()
    base_chars = sorted(set(corpus))

    show("一、训练一个 BPE")
    print(f"语料：{len(corpus)} 个字，{len(base_chars)} 种不同的字")
    print(f"最初词表 = 这些字本身：{len(base_chars)} 个")
    print()
    num_merges = 200
    merges = train_bpe(corpus, num_merges)
    byte_merges = train_byte_bpe(corpus, num_merges)
    print(f"做了 {num_merges} 次合并，词表变成 {len(base_chars) + num_merges} 个")
    print()
    print("前 20 次合并（学到的第一批 token）：")
    print("  " + "  ".join(left + right for left, right in merges[:20]))
    print()
    print("第 40-60 次合并（开始出现更长的片段）：")
    print("  " + "  ".join(left + right for left, right in merges[40:60]))
    print()

    show("二、用这个 BPE 来分词")
    print("下面几句都没有原样出现在语料里，有的连片段都没见过。")
    print()
    for sentence in ["苹果很甜", "苹果不好吃", "小米做成派",
                     "我昨天在商场看到苹果刚刚发布的新手机"]:
        chars = list(sentence)
        tokens = encode(sentence, merges)
        print(f"{sentence}")
        print(f"  字符级：{len(chars)} 个 token   {chars}")
        print(f"  BPE   ：{len(tokens)} 个 token   {tokens}")
    print()

    show("三、BPE 的两个好性质")
    sentence = "苹果发布新手机"
    tokens = encode(sentence, merges)
    print(f"性质一：能还原。decode(encode('{sentence}')) = "
          f"'{decode(tokens)}'")
    print("性质二：没有生词。BPE 最差的情况就是退回到一个个字符，")
    print("        而字符表是从语料里数出来的，永远够用。")
    print("        上面那个'不'字，在字符级词表里也只是一个普通的字。")
    print()

    show("四、同一份代码，换成字节就是 byte-level BPE")
    print("GPT-2 之后的模型用的是字节版。字节版的最小单位不是汉字，是 0-255 的字节。")
    print(f"'{sentence}' 一共 {len(sentence)} 个字，UTF-8 里是 {len(to_bytes(sentence))} 个字节")
    print()
    print(f"{'合并次数':<10}{'字符版 token 数':<18}{'字节版 token 数':<18}")
    print("-" * 64)
    for count in (0, 50, 200):
        char_tokens = token_count(sentence, merges[:count])
        byte_tokens = token_count(sentence, byte_merges[:count], unit=to_bytes)
        print(f"{count:<10}{char_tokens:<18}{byte_tokens:<18}")
    encoded = encode(sentence, byte_merges, unit=to_bytes)
    print()
    print(f"合并 200 次之后，字节版把这句话切成了："
          f"{[t.decode('utf-8', errors='replace') for t in encoded]}")
    print(f"还原回去：{from_bytes(encoded)}")
    print()
    print("看那两列：一开始字节版是字符版的三倍长（一个字三个字节），")
    print("合并多了以后能追上来 —— 最常见的合并就是'把一个字的三个字节粘成一个'。")
    print()
    print("字节版真正的好处：语料里从没出现过的字（繁体字、生僻字、表情）")
    print("                  也能用已经认识的字节拼出来，模型至少不会'完全没见过'。")


def train_byte_bpe(text, num_merges):
    """字节版 BPE —— 和字符版是同一个 train_bpe，只是最小单位换成了字节。"""
    return train_bpe(text, num_merges, unit=to_bytes)


if __name__ == "__main__":
    main()
