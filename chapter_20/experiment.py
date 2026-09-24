"""第 20 章 experiment：如果答案就在输入里，模型会怎么办？

三组实验，模型结构、窗口大小、训练步数、随机种子全都一样，
唯一的变化是：**输入的两个格子里分别放了什么。**

  A 组： [上一个字符, 当前字符]   ->  预测下一个字符     （诚实的做法）
  B 组： [上一个字符, 答案]       ->  预测下一个字符     （最后一格就是答案）
  C 组： [随机字符,   答案]       ->  预测下一个字符     （连上下文都是假的）

然后我们做一件事：给每个模型喂 33 个输入，每个输入的最后一格分别是 33 个字符中的一个，
看它的输出是不是正好等于最后一格。

如果输出就是最后一格，那它不是在预测，它是在**抄**。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, embedding, no_grad, randn, zeros, SGD

from after import (  # noqa: E402
    CHAR_TO_ID,
    CORPUS_IDS,
    DIM,
    ID_TO_CHAR,
    SEED,
    VOCAB,
    VOCAB_SIZE,
)


class WindowModel:
    """和第 20 章 after.py 里的模型同一个东西，窗口固定 2。"""

    def __init__(self, window=2, dim=DIM, vocab_size=VOCAB_SIZE, seed=SEED):
        self.window = window
        self.dim = dim
        self.table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.weight = randn(window * dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 1)
        self.bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        return [self.table, self.weight, self.bias]

    def __call__(self, windows):
        vectors = embedding(self.table, windows)
        flat = vectors.reshape(windows.shape[0], self.window * self.dim)
        return flat @ self.weight + self.bias


def build_pairs(ids, second_slot, random_context=False, seed=SEED):
    """造训练数据。

    second_slot = "current" -> 第二个格子放当前字符（诚实）
    second_slot = "answer"  -> 第二个格子放答案（偷看）
    random_context = True   -> 第一个格子换成随机字符
    """
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for position in range(1, len(ids) - 1):
        if random_context:
            first = int(rng.integers(0, VOCAB_SIZE))
        else:
            first = ids[position - 1]
        if second_slot == "answer":
            second = ids[position + 1]
        else:
            second = ids[position]
        inputs.append([first, second])
        targets.append(ids[position + 1])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=600, batch_size=64, learning_rate=0.5, seed=SEED):
    optimizer = SGD(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()


def evaluate(model, inputs, targets):
    with no_grad():
        logits = model(inputs)
        loss = cross_entropy(logits, targets).item()
        accuracy = float((logits.data.argmax(axis=-1) == targets).mean())
    return loss, accuracy


def run_group(second_slot, random_context=False):
    inputs, targets = build_pairs(CORPUS_IDS, second_slot, random_context)
    model = WindowModel(2)
    train(model, inputs, targets)
    loss, accuracy = evaluate(model, inputs, targets)
    return model, loss, accuracy


def copy_probe(model):
    """喂 33 个输入，最后一格分别是 33 个字符，看输出是不是等于最后一格。"""
    probe_ids = np.array([CHAR_TO_ID[char] for char in VOCAB])
    head = CHAR_TO_ID["低"]
    rows = np.array([[head, char_id] for char_id in probe_ids])
    with no_grad():
        predictions = model(rows).data.argmax(axis=-1)
    return probe_ids, predictions, float((predictions == probe_ids).mean())


def main():
    print("=" * 60)
    print("实验设置")
    print("=" * 60)
    print("  语料、模型结构（窗口 2）、训练步数、随机种子，三组完全一样。")
    print("  唯一变化的是输入的两个格子里放了什么：")
    print("    A 组  [上一个字符, 当前字符]")
    print("    B 组  [上一个字符, 答案]")
    print("    C 组  [随机字符,   答案]")
    print()

    groups = [
        ("A 组（最后一格是当前字符）", "current", False),
        ("B 组（最后一格是答案）", "answer", False),
        ("C 组（最后一格是答案，上下文随机）", "answer", True),
    ]

    results = []
    for name, second_slot, random_context in groups:
        model, loss, accuracy = run_group(second_slot, random_context)
        probe_ids, predictions, copy_rate = copy_probe(model)
        results.append((name, loss, accuracy, copy_rate, probe_ids, predictions))

    print("=" * 60)
    print("结果")
    print("=" * 60)
    print("  「抄答案」的意思是：输出正好等于输入的最后一格，")
    print("  在所有 33 个探测里占了多少。")
    print()
    for name, loss, accuracy, copy_rate, _, _ in results:
        print(f"  {name}")
        print(f"      loss {loss:.4f}      准确率 {accuracy:.2%}      抄答案 {copy_rate:.1%}")
        print()

    print("=" * 60)
    print("把最后一格换成别的字符，看它吐出什么")
    print("=" * 60)
    print("  探测方式：输入 = [低, c]，c 遍历 33 个字符。")
    print()
    for name, _, _, _, probe_ids, predictions in results:
        print(f"  {name}")
        for start in (0, 12, 24):
            chunk = list(zip(probe_ids[start:start + 12], predictions[start:start + 12]))
            pairs = "  ".join(f"{ID_TO_CHAR[c]}->{ID_TO_CHAR[p]}" for c, p in chunk)
            print(f"    {pairs}")
        print()

    print("=" * 60)
    print("这三组实验说明了什么")
    print("=" * 60)
    print("  1. B 组比 A 组 loss 低得多（上面的数字）。")
    print("     但 B 组并没有更会预测——它的输入里写着答案。")
    print()
    print("  2. C 组最能说明问题：它的上下文是完全随机的，")
    print("     一个正经的模型在 C 组的输入上应该什么也学不到。")
    print("     可它的准确率还是 100%，抄答案还是 100%。")
    print("     因为那个「100%」跟上下文一点关系都没有。")
    print()
    print("  3. 换句话说：**只要答案出现在输入里，模型就一定能找到它**，")
    print("     而且它会让 loss 变低，让你以为模型变好了。")
    print()
    print("  我们这个实验是故意摆出来的。下一章它会自己发生。")
    print("  attention 的作用是「让每个位置看到所有位置」。")
    print("  位置 t 要预测的是 t+1 这个字符，")
    print("  而 t+1 这个字符，就是位置 t+1 上的输入。")
    print("  到那时候，整段语料就变成了上面 B 组、C 组的样子。")


if __name__ == "__main__":
    main()
