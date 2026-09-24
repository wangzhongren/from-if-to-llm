"""第 21 章 after：把 attention 加回来，但这次挡住未来。

上一章 before.py 摆出了窗口拼接的两个毛病：长度写死、参数随窗口增长。

这一章我们把第 14 章的 attention 加回来。它是我们要的那个东西：
参数跟序列长度无关，每个位置自己决定要看哪里。

但是有一个坑。上一章最后我们已经看到了：

    输入 ids[t : t+T]，答案 ids[t+1 : t+T+1]
    位置 i 要预测的是 ids[t+i+1]

而 attention 会让位置 i 看到**所有**位置 —— 包括 ids[t+i+1]。

这一章做的事情就一件：**用一个掩码，把位置 i 后面的那些位置全部挡掉。**

挡掉的办法简单得有点好笑：把那些位置的分数填成 -1e9，
softmax 之后它们就变成 0 了。

    att = att.masked_fill(mask, -1e9)

这个掩码叫**因果掩码**（causal mask）。

after.py 把两种做法都训一遍，放在一起看。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Tensor, cross_entropy, embedding, no_grad, randn, zeros, SGD

# ---------------------------------------------------------------- 语料（和第 20 章一样）

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
ID_TO_CHAR = {i: char for char, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])

SEED = 20260924
DIM = 32        # 每个字符用几个数字表示
CONTEXT = 32    # 一次喂进去多少个字符
STEPS = 2000
LEARNING_RATE = 1.0


def causal_mask(length):
    """上三角（不含对角线）为 True 的方阵。

    mask[i][j] = True 的意思是「位置 i 不许看位置 j」。
    位置 i 只许看 j <= i，也就是只看自己和前面。
    """
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class AttentionLM:
    """embedding + 位置 + 单头 attention + 线性层。

    和 before.py 的窗口模型相比，换掉的只有一样东西：
    拼接窗口换成了 attention（第 14 章那个 Q / K / V）。
    """

    def __init__(self, vocab_size, dim=DIM, context=CONTEXT, seed=SEED):
        self.dim = dim
        self.context = context
        self.token_table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.position_table = randn(context, dim, scale=0.3, requires_grad=True, seed=seed + 1)
        self.query_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 2)
        self.key_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 3)
        self.value_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 4)
        self.head_weight = randn(dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 5)
        self.head_bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        return [
            self.token_table,
            self.position_table,
            self.query_weight,
            self.key_weight,
            self.value_weight,
            self.head_weight,
            self.head_bias,
        ]

    def __call__(self, token_ids, use_causal_mask=True):
        batch, length = token_ids.shape
        x = embedding(self.token_table, token_ids) + self.position_table[:length].reshape(1, length, self.dim)

        query = x @ self.query_weight
        key = x @ self.key_weight
        value = x @ self.value_weight

        # 第 14 章：Q 和 K 做点积，看「我在找的东西」和「你有什么」有多像
        attention = (query @ key.transpose(0, 2, 1)) * (1.0 / np.sqrt(self.dim))

        if use_causal_mask:
            # 把「我在找的东西」和「你有什么」之间的相似度，
            # 在位置 i 和它后面那些位置之间，强行抹掉
            attention = attention.masked_fill(causal_mask(length), -1e9)

        weights = attention.softmax(axis=-1)
        attended = weights @ value
        return attended @ self.head_weight + self.head_bias


# ---------------------------------------------------------------- 数据与训练


def build_batches(ids, context=CONTEXT):
    """整段语料切成一个个长度为 context 的窗口。

    每个窗口的输入是 ids[i:i+context]，答案是 ids[i+1:i+context+1]。
    位置 i 的答案，就是位置 i+1 的输入。这就是上一章说的那个陷阱。
    """
    inputs, targets = [], []
    for start in range(0, len(ids) - context - 1, context // 2):
        inputs.append(ids[start:start + context])
        targets.append(ids[start + 1:start + context + 1])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, use_causal_mask, steps=STEPS,
          batch_size=32, learning_rate=LEARNING_RATE, seed=SEED):
    optimizer = SGD(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs[batch], use_causal_mask), targets[batch])
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            history.append((step + 1, loss.item()))
    return history


def evaluate(model, inputs, targets, use_causal_mask):
    with no_grad():
        loss = cross_entropy(model(inputs, use_causal_mask), targets).item()
    return loss


def continue_text(model, prompt, length=30, use_causal_mask=True):
    """把模型的预测一个接一个接起来，看看能写出什么。

    （这个动作第 23 章会正式起个名字。现在先看结果。）
    """
    ids = [CHAR_TO_ID[char] for char in prompt]
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-model.context:])[None, :]
            logits = model(window, use_causal_mask)
            ids.append(int(logits.data[0, -1].argmax()))
    return "".join(ID_TO_CHAR[i] for i in ids)


# ---------------------------------------------------------------- 主流程


def main():
    inputs, targets = build_batches(CORPUS_IDS)
    print("=" * 60)
    print("语料与数据")
    print("=" * 60)
    print(f"  语料 {len(CORPUS)} 个字符，{VOCAB_SIZE} 个不同的字符")
    print(f"  切成 {len(inputs)} 个长度为 {CONTEXT} 的窗口")
    print("  每个窗口的输入是 ids[i : i+32]，答案是 ids[i+1 : i+33]")
    print("  **位置 i 的答案，就是位置 i+1 的输入。**")

    print()
    print("=" * 60)
    print("对照组一：不做任何遮挡")
    print("=" * 60)
    model_open = AttentionLM(VOCAB_SIZE)
    history_open = train(model_open, inputs, targets, use_causal_mask=False)
    for step, loss in history_open:
        print(f"  第 {step:>4} 步   loss {loss:.5f}")
    loss_open = evaluate(model_open, inputs, targets, use_causal_mask=False)
    print(f"  整个数据集上的 loss   {loss_open:.5f}")

    print()
    print("=" * 60)
    print("对照组二：把未来挡住（因果掩码）")
    print("=" * 60)
    model_causal = AttentionLM(VOCAB_SIZE)
    history_causal = train(model_causal, inputs, targets, use_causal_mask=True)
    for step, loss in history_causal:
        print(f"  第 {step:>4} 步   loss {loss:.5f}")
    loss_causal = evaluate(model_causal, inputs, targets, use_causal_mask=True)
    print(f"  整个数据集上的 loss   {loss_causal:.5f}")

    print()
    print("=" * 60)
    print("对比")
    print("=" * 60)
    print(f"  不做遮挡     loss {loss_open:.5f}")
    print(f"  因果掩码     loss {loss_causal:.5f}")
    print()
    print("  不做遮挡的那个，loss 低得多。看起来赢麻了。")
    print("  但 loss 只说明「它有多确信」，不说明「它学到的是什么」。")
    print("  让它们各自往下写一段看看。")

    print()
    print("=" * 60)
    print("把预测接起来")
    print("=" * 60)
    prompt = "床前明月光"
    print(f"  开头：{prompt}")
    print()
    print(f"  不做遮挡：{continue_text(model_open, prompt, use_causal_mask=False)}")
    print()
    print(f"  因果掩码：{continue_text(model_causal, prompt, use_causal_mask=True)}")

    print()
    print("=" * 60)
    print("为什么不做遮挡会是这样")
    print("=" * 60)
    print("  训练的时候，位置 i 的输入里放着位置 i+1 的字符。")
    print("  不加掩码，attention 就能看到它。")
    print("  所以模型学会了最简单的一招：**把下一个位置抄过来。**")
    print()
    print("  训练时它抄得到，loss 当然低。")
    print("  可是真到了要写字的时候，它写一个字，")
    print("  下一个位置原本是空的 —— 没得抄了。")
    print("  于是它退化成一个只会重复刚才见过的东西的机器。")
    print()
    print("  这两个模型的参数量一模一样，训练数据一模一样，")
    print("  差别只在 `attention.masked_fill(causal_mask(...), -1e9)` 这一行。")

    print()
    print("=" * 60)
    print("因果掩码挡住了什么")
    print("=" * 60)
    example = causal_mask(8)
    print("  一个长度 8 的因果掩码（True 表示「不许看」）：")
    print()
    print("        0  1  2  3  4  5  6  7     <- 被看的位置")
    for i in range(8):
        row = "  ".join("X" if example[i][j] else "." for j in range(8))
        print(f"   {i}   {row}    <- 看的位置 {i}")
    print()
    print("  X = 挡掉   . = 允许")
    print("  位置 0 谁也不看（除了自己），位置 7 谁都看。")
    print()
    print("  位置 i 看得见的位置，永远只有 0..i。")
    print("  这就是「因果」两个字的意思：")
    print("  一个位置只能被它之前的东西影响，不能反过来。")

    print()
    print("=" * 60)
    print("它解决了什么，还差什么")
    print("=" * 60)
    print("  解决了：")
    print("    - 序列多长都能吃（参数量跟长度无关）")
    print("    - 每个位置自己决定看哪里，不是固定往前看 8 格")
    print("    - 训练的时候不偷看答案了")
    print()
    print("  还差：")
    print("    这里只有一层 attention，后面直接就是线性层。")
    print("    第 19 章说过，attention 只会**交换**信息，不会**加工**信息；")
    print("    第 17、18 章说过，没有残差和 LayerNorm 就没法堆深。")
    print("    这些零件我们全都有，只是一直没装到一起。")
    print()
    print("    是时候了。")


if __name__ == "__main__":
    main()
