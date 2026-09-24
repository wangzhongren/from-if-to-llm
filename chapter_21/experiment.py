"""第 21 章 experiment：一步步把未来"放回来"，看模型什么时候开始作弊。

after.py 只给了两个端点：全挡、完全不挡。这里做一条连续的线。

我们给「位置 i 能看到多远」设一个数字 lookahead：

    lookahead = 0   位置 i 只能看 0..i     （因果掩码）
    lookahead = 1   位置 i 还能看 i+1      （而 i+1 那一格写的就是答案）
    lookahead = 3   位置 i 还能看 i+3
    lookahead = 7   位置 i 还能看 i+7
    lookahead = 999 谁都能看               （完全不挡）

对每一档，我们量三件事：

    训练 loss            它有多确信
    遮住未来之后的 loss    同一个模型，把未来挡上再测一遍
    接出来的文字          真让它写字的时候，写成了什么

中间那一列是关键。它专门用来拆穿「训练 loss 很低」这件事。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, embedding, no_grad, randn, zeros, SGD

from after import (  # noqa: E402
    AttentionLM,
    CHAR_TO_ID,
    CONTEXT,
    CORPUS_IDS,
    DIM,
    ID_TO_CHAR,
    LEARNING_RATE,
    SEED,
    STEPS,
    VOCAB_SIZE,
    build_batches,
)

LOOKAHEADS = (0, 1, 3, 7, 999)


def lookahead_mask(length, lookahead):
    """位置 i 不许看 j > i + lookahead。

    lookahead = 0 就是纯粹的因果掩码（只看自己和前面）。
    lookahead >= length 就什么都不挡。
    """
    rows = np.arange(length).reshape(-1, 1)
    columns = np.arange(length).reshape(1, -1)
    return columns > rows + lookahead


def forward(model, token_ids, mask):
    """和 after.py 里 AttentionLM.__call__ 一样，只是掩码由外面传进来。"""
    batch, length = token_ids.shape
    x = embedding(model.token_table, token_ids) + model.position_table[:length].reshape(1, length, DIM)
    query = x @ model.query_weight
    key = x @ model.key_weight
    value = x @ model.value_weight
    attention = (query @ key.transpose(0, 2, 1)) * (1.0 / np.sqrt(DIM))
    attention = attention.masked_fill(mask, -1e9)
    return (attention.softmax(axis=-1) @ value) @ model.head_weight + model.head_bias


def train_model(inputs, targets, lookahead, steps=STEPS,
                batch_size=32, learning_rate=LEARNING_RATE, seed=SEED):
    model = AttentionLM(VOCAB_SIZE)
    optimizer = SGD(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    mask = lookahead_mask(inputs.shape[1], lookahead)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        cross_entropy(forward(model, inputs[batch], mask), targets[batch]).backward()
        optimizer.step()
    return model


def loss_with(model, inputs, targets, lookahead):
    """用指定的掩码评估同一个模型。"""
    mask = lookahead_mask(inputs.shape[1], lookahead)
    with no_grad():
        return cross_entropy(forward(model, inputs, mask), targets).item()


def continue_text(model, lookahead, prompt, length=22):
    """把模型的预测一个接一个接起来。

    （这个动作第 23 章会正式起个名字。现在先看结果。）
    """
    ids = [CHAR_TO_ID[char] for char in prompt]
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-CONTEXT:])[None, :]
            mask = lookahead_mask(window.shape[1], lookahead)
            logits = forward(model, window, mask)
            ids.append(int(logits.data[0, -1].argmax()))
    return "".join(ID_TO_CHAR[i] for i in ids)


def describe(lookahead):
    if lookahead == 0:
        return "因果掩码"
    if lookahead > CONTEXT:
        return "不挡"
    return f"{lookahead} 格"


def main():
    inputs, targets = build_batches(CORPUS_IDS)
    print("=" * 60)
    print("实验：一点点把未来放回来")
    print("=" * 60)
    print(f"  训练数据：{len(inputs)} 个长度为 {CONTEXT} 的窗口")
    print("  每一档都用同样的模型、同样的数据、同样的步数，")
    print("  只改一个数字：位置 i 最多能往后看几格。")
    print()
    print("  「训练 loss」用它可以看的那一套掩码来量；")
    print("  「遮住未来之后的 loss」把同一个模型按最严的因果掩码再量一遍。")

    results = []
    for lookahead in LOOKAHEADS:
        model = train_model(inputs, targets, lookahead)
        results.append((
            lookahead,
            loss_with(model, inputs, targets, lookahead),
            loss_with(model, inputs, targets, 0),
            continue_text(model, lookahead, "床前明月光"),
        ))

    print()
    print("=" * 60)
    print("结果")
    print("=" * 60)
    print("  往后看       训练 loss   遮住未来之后   接出来的文字")
    print("  " + "-" * 62)
    for lookahead, train_loss, honest_loss, text in results:
        print(f"  {describe(lookahead):>8}   {train_loss:>9.5f}   {honest_loss:>10.5f}   {text}")
    print()

    print("=" * 60)
    print("读这三列")
    print("=" * 60)
    print("  第一列（训练 loss）：四档几乎一样低。")
    print("  因果掩码那一档反而是**最高**的。")
    print("  如果你只看这一列，你会得出结论：")
    print("  「加掩码是在拖后腿，让它多看一点总是好的。」")
    print()
    print("  第二列（遮住未来之后）：因果掩码那一档纹丝不动，")
    print("  其他四档全部爆炸 —— 从 0.0005 涨到 5 左右。")
    print("  同一个模型，同一个数据集，只是把未来挡上，就完全不会了。")
    print()
    print("  原因是：它从来没有学会预测。")
    print("  它学会的是「往右边那一格看一眼，抄过来」。")
    print("  训练的时候右边那一格一直都在，所以它抄得又快又准。")
    print()
    print("  第三列（接出来的文字）：写出来是什么，一眼就看得出来。")
    print()
    print("  一句话：**训练 loss 低，不代表模型学会了你想让它学的东西。**")
    print("  想验证它到底会不会，就得把它依赖的那条捷径掐掉，再测一遍。")


if __name__ == "__main__":
    main()
