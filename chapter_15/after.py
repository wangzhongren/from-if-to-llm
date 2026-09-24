"""第 15 章 after.py —— 多头注意力。

第 14 章一个词只有一行权重。这一章我们让同一个词同时有好几行：

    每个头有自己的 W_q / W_k / W_v，各自算一套权重，
    各自把 value 混一遍，最后把几个头的结果拼起来。

一句话里同时有好几种关系（语法、指代、时间……），
一种"看的方式"不够，那就并行几种。

这个文件还会真的训练它：用上下文猜每个词是谁，看几个头会不会长出不一样的模式。
（数据只有两句话，几秒钟就训完。）

跑法：

    ./.venv/bin/python chapter_15/after.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import SGD, Tensor, cross_entropy, randn

# ---------------------------------------------------------------- 语料

SENTENCE_1 = "我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机".split()
SENTENCE_2 = "小王 把 书 给了 小李 因为 他 明天 考试".split()
SENTENCES = [SENTENCE_1, SENTENCE_2]

# 8 个维度依次代表：
#   人/动物   时间   地点   动作   物品   科技与商业   虚词   程度
BASE_EMBEDDINGS = {
    "我":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    "昨天": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    "在":   [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.8, 0.0],
    "商场": [0.0, 0.0, 1.0, 0.0, 0.3, 0.2, 0.0, 0.0],
    "看到": [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
    "苹果": [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.0, 0.0],
    "刚刚": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.7],
    "发布": [0.0, 0.0, 0.0, 0.8, 0.3, 1.0, 0.0, 0.0],
    "的":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "新":   [0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.0, 1.0],
    "手机": [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.0, 0.0],
    "小王": [0.9, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
    "把":   [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 1.0, 0.0],
    "书":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "给了": [0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0],
    "小李": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    "因为": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.9, 0.3],
    "他":   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    "明天": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    "考试": [0.0, 0.5, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
}

VOCAB = list(BASE_EMBEDDINGS.keys())
WORD_ID = {word: i for i, word in enumerate(VOCAB)}

DIM = 24        # 词向量维度
NUM_HEADS = 4   # 头的个数
HEAD_DIM = DIM // NUM_HEADS


# ---------------------------------------------------------------- 参数


def make_parameters(seed=15, num_heads=NUM_HEADS):
    """初始化一套参数。

    词向量表不是从随机数开始的：我们把第 13 章那张 8 维的表投到 24 维当起点，
    然后让它跟着训练一起更新。
    """
    head_dim = DIM // num_heads
    base = np.array([BASE_EMBEDDINGS[word] for word in VOCAB])
    projector = randn(8, DIM, scale=0.5, seed=seed)
    table = Tensor(base @ projector.data, requires_grad=True)

    return {
        "num_heads": num_heads,
        "table": table,
        # 每个头都有自己的三个投影矩阵：形状 (头数, 输入维度, 每个头的维度)
        "w_q": randn(num_heads, DIM, head_dim, scale=0.5, requires_grad=True, seed=seed + 1),
        "w_k": randn(num_heads, DIM, head_dim, scale=0.5, requires_grad=True, seed=seed + 2),
        "w_v": randn(num_heads, DIM, head_dim, scale=0.5, requires_grad=True, seed=seed + 3),
        # 几个头拼起来之后，再投一次影，让它们的信息互相混一下
        "w_o": randn(DIM, DIM, scale=0.5, requires_grad=True, seed=seed + 4),
    }


def all_parameters(params):
    return [params["table"], params["w_q"], params["w_k"], params["w_v"], params["w_o"]]


# ---------------------------------------------------------------- 前向


def multi_head_attention(vectors, params):
    """多头注意力的前向。

    vectors : (词数, DIM)
    返回    : (输出 (词数, DIM), 每个头的权重 (头数, 词数, 词数))
    """
    num_words = vectors.shape[0]
    head_dim = params["w_q"].shape[-1]

    # 三个投影一次算完所有头：矩阵乘法会自动在"头"这一维上广播
    query = vectors @ params["w_q"]          # (头数, 词数, HEAD_DIM)
    key = vectors @ params["w_k"]
    value = vectors @ params["w_v"]

    # 每个头自己算一套分数、一套权重
    scores = (query @ key.transpose(0, 2, 1)) / np.sqrt(head_dim)
    weights = scores.softmax(axis=-1)        # (头数, 词数, 词数)

    # 每个头自己混一遍 value
    head_output = weights @ value            # (头数, 词数, HEAD_DIM)

    # 把几个头拼起来：(头数, 词数, HEAD_DIM) -> (词数, 头数, HEAD_DIM) -> (词数, DIM)
    joined = head_output.transpose(1, 0, 2).reshape(num_words, DIM)
    return joined @ params["w_o"], weights


def sentence_loss(sentence, params):
    """用上下文猜每个词是谁，返回这一句的平均损失。"""
    ids = [WORD_ID[word] for word in sentence]
    vectors = params["table"][ids]
    output, _ = multi_head_attention(vectors, params)
    logits = output @ params["table"].T      # (词数, 词表大小)
    return cross_entropy(logits, ids)


def train(steps=400, lr=0.1, seed=15, num_heads=NUM_HEADS):
    params = make_parameters(seed, num_heads)
    optimizer = SGD(all_parameters(params), lr=lr)
    history = []
    for step in range(steps + 1):
        optimizer.zero_grad()
        loss = Tensor(0.0, requires_grad=True)
        for sentence in SENTENCES:
            loss = loss + sentence_loss(sentence, params)
        loss = loss * (1.0 / len(SENTENCES))
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return params, history


# ---------------------------------------------------------------- 打印

HEAT = ".:-=+*#%@"


def display_width(text):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def pad(text, width, align="left"):
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def heat_row(values, full_scale):
    """把一行权重画成一串热力字符：越重越"黑"。"""
    out = ""
    for value in values:
        level = int(value / full_scale * (len(HEAT) - 1))
        out += HEAT[level] * 4
    return out


def show_heads(words, weights, title):
    """把每个头的权重矩阵画成热力字符。"""
    print(f"\n{title}")
    print("  热力字符：'.' 最小，'@' 最大。每个头按自己的最大值画，")
    print("  所以这张图看的是'形状'——这个头在关注谁；精确数字在下一节。")
    for head in range(weights.shape[0]):
        matrix = weights[head]
        full_scale = float(matrix.max())
        print(f"\n  【头 {head}】（本头的最大值 = {full_scale:.3f}）")
        print("    " + pad("", 6) + "".join(pad(word, 4) for word in words))
        for i, word in enumerate(words):
            print("    " + pad(word, 6) + heat_row(matrix[i], full_scale))


def show_row_bars(words, weights, position, title):
    print(f"\n{title}")
    for head in range(weights.shape[0]):
        row = weights[head][position]
        print(f"\n  头 {head}：")
        for j, word in enumerate(words):
            mark = "  <- 自己" if j == position else ""
            print(f"    {pad(word, 6)}{row[j]:.4f}  {'#' * int(round(row[j] * 40))}{mark}")


def main():
    print("=" * 60)
    print("语料：")
    for sentence in SENTENCES:
        print("  " + " ".join(sentence))
    print(f"词表大小 {len(VOCAB)}，词向量 {DIM} 维，{NUM_HEADS} 个头，"
          f"每个头 {HEAD_DIM} 维")
    print("=" * 60)

    params, history = train(steps=400)
    print("\n训练（用上下文猜每个词是谁）：")
    for step in range(0, len(history), 100):
        print(f"  第 {step:>3} 步   损失 = {history[step]:.4f}")
    print(f"  第 {len(history) - 1:>3} 步   损失 = {history[-1]:.4f}")

    # ---------------- 每个头看到了什么 ----------------
    words = SENTENCE_2
    ids = [WORD_ID[word] for word in words]
    _, weights = multi_head_attention(params["table"][ids], params)
    weights = weights.data

    show_heads(words, weights, "训练之后，每个头的注意力矩阵（句子：小王 把 书 给了 小李 因为 他 明天 考试）")

    show_row_bars(words, weights, words.index("他"), "只看'他'这个字：4 个头分别在问谁")

    print("\n" + "=" * 60)
    print("同一个位置，4 个头给出 4 套不同的权重")
    print("=" * 60)
    for head in range(NUM_HEADS):
        row = weights[head][words.index("他")]
        top = sorted(range(len(words)), key=lambda j: -row[j])[:3]
        best = "、".join(f"{words[j]}({row[j]:.3f})" for j in top)
        print(f"  头 {head}：{best}")
    print("\n  如果只有一个头，'他'就只能有一套权重——"
          "上面这 4 种看的方式，只能留一种。")


if __name__ == "__main__":
    main()
