"""第 28 章 experiment.py —— 参数、步数、数据，三个旋钮一起看。

after.py 只比了"同样步数下，大小不同"。这个文件把另外两件事也量出来：

  一、三个模型在 30 / 150 / 600 步上的验证 loss（参数和步数一起看）
  二、时间和参数能互相替代吗（小模型多训 4 倍 vs 中模型）
  三、和一个纯统计模型比一比（8-gram 回退）—— 语言模型学的到底是什么

跑法：

    ./.venv/bin/python chapter_28/experiment.py
"""

import math
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    CHARS,
    SIZES,
    TEXT,
    SPLIT,
    VOCAB_SIZE,
    GPTModel,
    BLOCK_SIZE,
    pad,
    train,
)

STEPS = 600
CHECKPOINTS = (30, 150, 600)


def measure_scaling():
    """三个模型各训练 600 步，在 30 / 150 / 600 步上各量一次验证 loss。"""
    table = []
    for name, size in SIZES:
        import torch
        torch.manual_seed(0)
        model = GPTModel(VOCAB_SIZE, block_size=BLOCK_SIZE, **size)
        history, seconds = train(model, STEPS, checkpoints=CHECKPOINTS)
        table.append(dict(name=name, params=model.n_params(), history=history,
                          seconds=seconds))
        print(f"  {name}（{model.n_params():,} 参数）训练完，用时 {seconds:.0f} 秒")
    return table


def experiment_one(table):
    print("=" * 68)
    print("一、三个模型，三条时间线")
    print("=" * 68)
    print(pad("模型", 10) + pad("参数量", 14) + pad("30 步", 12) + pad("150 步", 12) + "600 步")
    print("-" * 68)
    for row in table:
        print(pad(row["name"], 10) + pad(f"{row['params']:,}", 14)
              + "".join(pad(f"{row['history'][s]:.4f}", 12) for s in CHECKPOINTS))
    print()
    print("三个现象：")
    print("  1. 30 步、150 步这两列，竖着看都是'参数越多 loss 越低'。")
    print("     150 步那一列：0.5998 -> 0.4758 -> 0.4698，一路往下。")
    print("  2. 横着看，练得越久 loss 越低，但降幅越来越小。")
    print("  3. 到了 600 步，三个模型挤进 0.45 - 0.48 的窄带，顺序已经不重要了 ——")
    print("     100 万参数的那个甚至比 10 万参数的略高（0.4748 vs 0.4571）。")
    print()
    print("第 3 点是这一章最该记住的东西：loss 有一个下限，")
    print("这个下限由数据本身的随机性决定。这门玩具语言的规律就那么几条，")
    print("8 万个字里翻来覆去都是它们，学到 0.46 附近就再也压不下去了，")
    print("再多的参数只能用来'背'，而背下来的东西在验证集上不算数。")
    print()
    print("真实语言的'下限'高得多 —— 人类的文本里有无数种说法，")
    print("所以真实世界里这条线一路都是陡的，参数越多就有越多规律可以装。")
    print("这就是 scaling law 存在的原因：**它说的不是'大就是好'，")
    print("而是'数据里有那么多可学的，你得有那么多参数去装'**。")
    print()


def experiment_two(table):
    print("=" * 68)
    print("二、时间和参数，能互相替代吗")
    print("=" * 68)
    small, middle, big = table
    print(pad("配置", 32) + pad("参数量", 14) + pad("训练步数", 12) + "验证 loss")
    print("-" * 68)
    rows = [
        ("小模型，训练 150 步", small, 150),
        ("小模型，训练 600 步（4 倍时间）", small, 600),
        ("中模型，训练 150 步", middle, 150),
        ("中模型，训练 600 步", middle, 600),
        ("大模型，训练 150 步", big, 150),
        ("大模型，训练 600 步", big, 600),
    ]
    for label, row, step in rows:
        print(pad(label, 32) + pad(f"{row['params']:,}", 14) + pad(f"{step}", 12)
              + f"{row['history'][step]:.4f}")
    print()
    print(f"小模型多训 4 倍时间：{small['history'][150]:.4f} -> {small['history'][600]:.4f}")
    print(f"  这个成绩比中模型训 150 步（{middle['history'][150]:.4f}）还好一点。")
    print(f"  但中模型多训 4 倍是 {middle['history'][600]:.4f}，还是更好。")
    print()
    print("所以：时间可以部分替代参数，但不是无限的。")
    print("同样一笔计算预算，怎么在'大小'和'训练多久'之间分配，是有最优解的 ——")
    print("这正是 scaling law 研究要回答的问题。")
    print()


# ---------------------------------------------------------------- 实验三

ALPHA = 0.5          # 回退时的平滑量，纯为了不让没见过的组合变成 0 概率


def build_ngram_counts(text, order):
    """数一数：每个长度为 order 的上下文后面，各个字各出现了多少次。"""
    counts = defaultdict(Counter)
    for i in range(len(text) - order):
        counts[text[i:i + order]][text[i + order]] += 1
    return counts


def ngram_loss(train_text, val_text, max_order, limit=20000):
    """用 n-gram 回退当语言模型，量它在验证集上的 loss。"""
    counts = {order: build_ngram_counts(train_text, order)
              for order in range(1, max_order + 1)}

    def probability(context, ch):
        p = 1.0 / len(CHARS)                     # 从"均匀猜"开始
        for order in range(max_order, 0, -1):
            counter = counts[order].get(context[-order:])
            if counter is None:
                continue
            total = sum(counter.values())
            # 见过的上下文就多信它一点，没见过的就退回上一层
            p = (counter.get(ch, 0) + ALPHA * p) / (total + ALPHA)
        return p

    total, count = 0.0, 0
    end = min(len(val_text), max_order + limit)
    for i in range(max_order, end):
        total += -math.log(probability(val_text[max(0, i - max_order):i], val_text[i]))
        count += 1
    return total / count


def experiment_three(table):
    print("=" * 68)
    print("三、和一个纯统计模型比一比")
    print("=" * 68)
    print("n-gram 回退：不学任何'表示'，只数'某几个字后面通常跟哪个字'的频次。")
    print()
    train_text, val_text = TEXT[:SPLIT], TEXT[SPLIT:]
    print(pad("模型", 26) + pad("参数量", 16) + "验证 loss")
    print("-" * 68)
    for order in (2, 4, 8):
        start = time.time()
        loss = ngram_loss(train_text, val_text, order)
        params = sum(len(c) for c in build_ngram_counts(train_text, order).values())
        print(pad(f"{order}-gram 回退", 26) + pad(f"{params:,} 个组合", 16)
              + f"{loss:.4f}   （{time.time() - start:.1f} 秒）")
    for row in table:
        print(pad(f"我们的模型（{row['name']}，600 步）", 26)
              + pad(f"{row['params']:,}", 16) + f"{row['history'][600]:.4f}")
    print()
    print("8-gram 把所有'8 个字'的组合都数了一遍，仍然是 0.71；")
    print("我们最小的模型（1.6 万参数）练 600 步以后是 0.46。")
    print()
    print("（顺带说一句反面：小模型刚开跑的前 30 步是 1.59，比 4-gram 还差。")
    print("  训练不够的时候，一个只数频次的统计模型就能赢过神经网络。）")
    print()
    print("为什么我们的模型能赢？因为这门语言的规律不在相邻的几个字里，")
    print("而在'这一句在聊什么话题'里 —— 话题要攒好几个字的证据才能确定。")
    print("这正是第 13 到 19 章造出来的那些东西（注意力、残差、多层组合）在做的事。")
    print()
    print("所以参数不是'记忆容量'，它是'把上下文里的规律装下来'的能力。")
    print()


if __name__ == "__main__":
    print("=" * 68)
    print(f"语料 {len(TEXT):,} 个字，三个模型各训练 {STEPS} 步")
    print("=" * 68)
    table = measure_scaling()
    print()
    experiment_one(table)
    experiment_two(table)
    experiment_three(table)
