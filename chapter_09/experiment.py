"""这一章的实验：让自动求导和数值法当面对质。

自动求导说：我知道每个参数的梯度是多少。
我们凭什么信它？

凭第 6 章那个笨办法：把参数挪动一丁点，看 loss 怎么变。
那个办法慢，但它绝对老实 —— 它连"梯度"是什么都不用知道。

两边算出来的数字如果一模一样，那自动求导就是对的。

后面还做三件事：
  - 看看 no_grad() 到底省掉了什么；
  - 看看 backward() 为什么要先 zero_grad()；
  - 把这个引擎用回第 6 章那 10 句话。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from toygrad import SGD, Tensor, cross_entropy, no_grad, randn, zeros  # noqa: E402

# XOR 的 4 个点：两个数一样就是 1（同类），不一样就是 0
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Y = np.array([1, 0, 0, 1])


def make_params(seed=0):
    """和第 8 章一样的网络：2 -> 4（relu）-> 2。

    偏置也给一点小的随机值。为什么不全填 0？
    因为全填 0 的话，第一个样本传进来的中间层输出正好是 0，
    而 0 就是 relu 的折点。数值法踩在折点上会给出没有意义的数 ——
    那时候两边对不上，是数值法的问题，不是引擎的问题。
    """
    return [
        randn(2, 4, scale=0.5, requires_grad=True, seed=seed),
        randn(4, scale=0.1, requires_grad=True, seed=seed + 5),
        randn(4, 2, scale=0.5, requires_grad=True, seed=seed + 1),
        randn(2, scale=0.1, requires_grad=True, seed=seed + 6),
    ]


def model_loss(params, x_matrix, targets):
    """一次算完全部 4 个样本。数组版的好处在这里最明显。"""
    hidden = (Tensor(x_matrix) @ params[0] + params[1]).relu()
    logits = hidden @ params[2] + params[3]
    return cross_entropy(logits, targets)


def numeric_gradients(params, x_matrix, targets, h=1e-6):
    """第 6 章的数值法。

    它只认识 loss 这一个数字，完全不知道里面有 relu、有 softmax、
    有几层。所以它可以给任何东西当裁判。
    """
    grads = []
    for p in params:
        flat = p.data.reshape(-1).copy()
        out = np.zeros_like(flat)
        for i in range(flat.size):
            original = flat[i]

            flat[i] = original + h
            p.data = flat.reshape(p.data.shape)
            plus = model_loss(params, x_matrix, targets).data

            flat[i] = original - h
            p.data = flat.reshape(p.data.shape)
            minus = model_loss(params, x_matrix, targets).data

            flat[i] = original
            p.data = flat.reshape(p.data.shape)
            out[i] = (plus - minus) / (2 * h)

        p.data = flat.reshape(p.data.shape)
        grads.append(out.reshape(p.data.shape))
    return grads


def count_graph_nodes(tensor):
    """数一张计算图上有多少个节点。"""
    seen = set()
    stack = [tensor]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node._prev)
    return len(seen)


# ------------------------------------------------------------------ 第 1 步


def step1_face_off():
    print("=" * 60)
    print("第 1 步：自动求导 vs 数值法")
    print("=" * 60)
    print("网络：2 -> 4（relu）-> 2，一共 22 个参数。")
    print()

    params = make_params(seed=0)
    for p in params:
        p.zero_grad()

    loss = model_loss(params, X, Y)
    loss.backward()                       # <- 自动求导算出来的
    numeric = numeric_gradients(params, X, Y)

    labels = ["w1", "b1", "w2", "b2"]
    rows = []
    labels_of = []
    for name, p, num in zip(labels, params, numeric):
        flat_auto = p.grad.reshape(-1)
        flat_num = num.reshape(-1)
        for i in range(flat_auto.size):
            rows.append((flat_auto[i], flat_num[i], abs(flat_auto[i] - flat_num[i])))
            labels_of.append(f"{name}[{i}]")

    # 挑绝对值最大的几个来展示（有些中间单元被 relu 掐死了，梯度本来就是 0）
    order = sorted(range(len(rows)), key=lambda k: -abs(rows[k][0]))[:8]
    print(f"{'参数':>12}{'自动求导':>18}{'数值法':>18}{'差':>12}")
    for k in order:
        auto_value, num_value, diff = rows[k]
        print(f"{labels_of[k]:>12}{auto_value:>18.10f}{num_value:>18.10f}"
              f"{diff:>12.2e}")

    worst = max(row[2] for row in rows)
    total = sum(p.grad.size for p in params)
    print(f"（表里只列了 8 个，一共比了 {total} 个参数）")
    print()
    print(f"全部 {total} 个参数里，最大的差是 {worst:.2e}。")
    print("这不是「差不多」，这是同一个数字的两种算法。")
    print()


# ------------------------------------------------------------------ 第 2 步


def step2_training():
    print("=" * 60)
    print("第 2 步：用它把 XOR 训出来")
    print("=" * 60)

    params = make_params(seed=0)
    optimizer = SGD(params, lr=0.5)

    print(f"{'轮数':>4}{'loss':>16}")
    for step in range(301):
        optimizer.zero_grad()             # <- 每轮都要清零，原因见第 4 步
        loss = model_loss(params, X, Y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"{step:>4}{loss.item():>16.10f}")

    hidden = (Tensor(X) @ params[0] + params[1]).relu()
    logits = (hidden @ params[2] + params[3]).data
    guess = (logits[:, 0] < logits[:, 1]).astype(int)
    print()
    print(f"准确率：{np.mean(guess == Y):.2f}")
    print()
    print(f"{'输入':>10}{'答案':>6}{'两个分数':>26}{'判成':>6}")
    for i in range(4):
        print(f"{str([int(v) for v in X[i]]):>10}{Y[i]:>6}"
              f"{logits[i, 0]:>+12.3f} /{logits[i, 1]:>+12.3f}{guess[i]:>6}")
    print()


# ------------------------------------------------------------------ 第 3 步


def step3_no_grad():
    print("=" * 60)
    print("第 3 步：no_grad() 省掉了什么")
    print("=" * 60)
    print("每一次运算，引擎都要记一笔「我是怎么算出来的」。")
    print("这些记录堆起来就是计算图。训练的时候必须记 ——")
    print("不然 backward() 就不知道该往回传给谁。")
    print()
    print("但推理的时候不需要。我们只是想拿一个结果，不想更新任何东西。")
    print()

    params = make_params(seed=0)

    out = model_loss(params, X, Y)
    print(f"正常算一遍：这张图上有 {count_graph_nodes(out)} 个节点，"
          f"requires_grad = {out.requires_grad}")

    with no_grad():
        out_quiet = model_loss(params, X, Y)
    print(f"用 no_grad()：这张图上有 {count_graph_nodes(out_quiet)} 个节点，"
          f"requires_grad = {out_quiet.requires_grad}")
    print()
    print("两种算法给出的 loss 一模一样：")
    print(f"  {out.item():.10f}")
    print(f"  {out_quiet.item():.10f}")
    print()
    print("区别只有一个：第一种一直在记账，第二种什么都没记。")
    print("第 23 章模型生成文字的时候，用的就是第二种。")
    print()


# ------------------------------------------------------------------ 第 4 步


def step4_accumulate():
    print("=" * 60)
    print("第 4 步：为什么每个参数的梯度都是「加」上去的")
    print("=" * 60)
    print("一个参数如果在图里被用了两次（比如 a * a），")
    print("那它就对结果有两条影响路径，两条路上的梯度必须加起来。")
    print("所以每个 _backward 里写的都是「+=」，不是「=」。")
    print()

    a = randn(4, requires_grad=True, seed=0)
    loss = (a * a).sum()
    loss.backward()
    print(f"a 的值           : {np.round(a.data, 4)}")
    print(f"一次 backward 之后: {np.round(a.grad, 4)}   （正好是 2a）")

    loss = (a * a).sum()
    loss.backward()
    print(f"再来一次 backward : {np.round(a.grad, 4)}   （变成了 4a）")
    print()

    a.zero_grad()
    print(f"zero_grad() 之后  : {np.round(a.grad, 4)}")
    print()
    print("第二次算的是同一个 loss，梯度却翻了一倍 —— 因为它累加了，")
    print("而不是被覆盖。这和 PyTorch 的行为一模一样。")
    print()
    print("为什么要把「+=」设计成默认？因为一个参数被用两次是家常便饭：")
    print("比如同一个词的权重在一个句子里被用了两次（同一个词出现两次），")
    print("那两条路上的梯度就得加起来。")
    print()
    print("代价就是：每一轮训练之前必须手动清零。")
    print("忘了清零，梯度会一轮轮累积，训练会越来越疯。")
    print()


# ------------------------------------------------------------------ 第 5 步


def step5_back_to_sentences():
    print("=" * 60)
    print("第 5 步：把这个引擎用回第 6 章那 10 句话")
    print("=" * 60)

    vocab = ["苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
             "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派"]
    word_to_id = {word: i for i, word in enumerate(vocab)}
    sentences = ["苹果发布新手机", "苹果发布新芯片", "华为发布新电脑",
                 "小米发布新手机", "苹果芯片很强", "苹果很好吃", "苹果很甜",
                 "香蕉很好吃", "这个苹果真甜", "苹果做成派"]
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def tokenize(sentence):
        tokens, i = [], 0
        while i < len(sentence):
            for length in (2, 1):
                piece = sentence[i:i + length]
                if piece in word_to_id:
                    tokens.append(piece)
                    i += length
                    break
            else:
                i += 1
        return tokens

    def featurize(sentence):
        x = [0.0] * len(vocab)
        for token in tokenize(sentence):
            x[word_to_id[token]] += 1.0
        return x

    features = np.array([featurize(s) for s in sentences])
    print(f"输入：每句话变成一个 {len(vocab)} 维的计数向量。")

    w = randn(len(vocab), 2, scale=0.1, requires_grad=True, seed=0)
    b = zeros(2, requires_grad=True)
    optimizer = SGD([w, b], lr=0.5)

    first = last = None
    for step in range(200):
        optimizer.zero_grad()
        loss = cross_entropy(Tensor(features) @ w + b, labels)
        if step == 0:
            first = loss.item()
        last = loss.item()
        loss.backward()
        optimizer.step()

    logits = ((Tensor(features) @ w + b)).data
    correct = int(np.sum(logits.argmax(axis=1) == labels))
    print(f"训练 200 步：loss {first:.6f} -> {last:.8f}，"
          f"10 句里对了 {correct} 句。")
    print()

    print("它学到的权重（科技那一类）：")
    for i, word in enumerate(vocab):
        print(f"  {word:>4}  第 {i:>2} 号  权重 {w.data[i, 0]:>+7.3f}")
    print()
    print("权重是我们见过的：只出现在科技句里的词，权重是正的；")
    print("只出现在食品句里的词，权重是负的。")
    print()
    print("但请注意每一行最左边那两个数字：**第 0 号、第 1 号、第 2 号**。")
    print("词表里的这 16 个位置，是我们在第 1 章随手排的。")
    print("「苹果」是第 0 号，「香蕉」是第 11 号 ——")
    print("这两个编号告诉我们什么了吗？")
    print("没有。它们只是编号。")
    print()
    print("而这个模型看到的，就是这 16 个位置上的计数。")
    print("在它眼里，「苹果」和「香蕉」之间的差别，")
    print("和「第 0 号位置」与「第 11 号位置」之间的差别，是一回事。")


def main():
    step1_face_off()
    step2_training()
    step3_no_grad()
    step4_accumulate()
    step5_back_to_sentences()


if __name__ == "__main__":
    main()
