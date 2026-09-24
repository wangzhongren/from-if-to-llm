"""第 14 章 experiment.py —— 那个 √d 到底是干什么的。

Q 和 K 的点积有多大，取决于它们的维度 d：维度越高，加起来的东西越多，分数就越大。
分数一大，softmax 就变成了"只挑一个"，而且权重对分数不再敏感——也就是学不动。

这个文件用真实的数字把这件事量出来：

  表格：维度 d 从 8 到 512，看不缩放和除以 √d 之后，分数和权重各是什么样。
  例子：d = 512 时"苹果"那一行，权重被压成了 1.00000000 和十个 0.00000000。

跑法：

    ./.venv/bin/python chapter_14/experiment.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import CENTER, SENTENCE, pad, vectors_of  # noqa: E402
from toygrad import Tensor  # noqa: E402

DIMS = [8, 16, 32, 64, 128, 256, 512]
REPEATS = 20          # 每档随机初始化 20 次，取平均，免得被一次运气带偏


def softmax_row(scores):
    """一维 softmax。"""
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def effective_count(weights):
    """有效词数：注意力平均摊在几个词上。

    每一行权重都一样的极端情形是"摊在 11 个词上"，这个数是 11；
    只盯着一个词看的时候，这个数是 1。
    """
    return float(np.exp(-(weights * np.log(weights + 1e-12)).sum()))


def sensitivity(weights):
    """一行权重对分数的敏感度，也就是反向传播要用的那个梯度。

    softmax 的导数是一个矩阵：diag(p) - p pᵀ。这里取它的范数。
    权重被压成"1 和一堆 0"之后，这个数会变成 0——
    意思是：分数再怎么动，权重也不动，参数也就学不动了。
    """
    jacobian = np.diag(weights) - np.outer(weights, weights)
    return float(np.linalg.norm(jacobian))


def make_scores(dim, seed):
    """随机初始化两组投影，算出"苹果"那一行的分数。"""
    matrix = vectors_of(SENTENCE).data
    rng = np.random.default_rng(seed)
    w_q = rng.standard_normal((matrix.shape[-1], dim))
    w_k = rng.standard_normal((matrix.shape[-1], dim))
    scores = (matrix @ w_q) @ (matrix @ w_k).T
    return scores[CENTER]


def main():
    print("=" * 74)
    print("实验：维度和分数的关系（'苹果'那一行的平均表现，每组随机初始化 20 次）")
    print("=" * 74)
    print(f"\n  {'':>4}  |  {'分数标准差':^17}  |  {'最大权重':^17}  |  {'有效词数':^17}")
    print(f"  {'d':>4}  |  {'不缩放':>7}{'除以√d':>10}  |  {'不缩放':>7}{'除以√d':>10}  "
          f"|  {'不缩放':>7}{'除以√d':>10}")
    print("  " + "-" * 70)

    rows = []
    for dim in DIMS:
        total = np.zeros(6)
        for t in range(REPEATS):
            scores = make_scores(dim, 1000 * t + dim)
            scaled = scores / np.sqrt(dim)
            p_plain = softmax_row(scores)
            p_scaled = softmax_row(scaled)
            total += [
                scores.std(), scaled.std(),
                p_plain.max(), p_scaled.max(),
                effective_count(p_plain), effective_count(p_scaled),
            ]
        total /= REPEATS
        rows.append((dim, total))
        print(f"  {dim:>4}  |  {total[0]:>7.2f}{total[1]:>10.2f}  |  "
              f"{total[2]:>7.3f}{total[3]:>10.3f}  |  {total[4]:>7.2f}{total[5]:>10.2f}")

    plain_std = [r[1][0] for r in rows]
    scaled_std = [r[1][1] for r in rows]
    plain_eff = [r[1][4] for r in rows]
    scaled_eff = [r[1][5] for r in rows]
    print(f"\n  不缩放：分数标准差从 {plain_std[0]:.2f} 一路涨到 {plain_std[-1]:.2f}，"
          f"有效词数从 {plain_eff[0]:.2f} 掉到 {plain_eff[-1]:.2f}。")
    print(f"  除以 √d：无论维度多高，分数标准差都在 "
          f"{min(scaled_std):.2f}~{max(scaled_std):.2f} 之间，"
          f"有效词数都在 {min(scaled_eff):.2f}~{max(scaled_eff):.2f} 之间。")

    # ---------------- 梯度那一栏 ----------------
    print("\n" + "=" * 74)
    print("再加一栏：权重对分数的敏感度（这就是反向传播要用的梯度）")
    print("=" * 74)
    print(f"\n  {'d':>4}  |  {'不缩放：最大权重':>16}{'梯度':>12}  |  "
          f"{'除以√d：最大权重':>16}{'梯度':>12}")
    print("  " + "-" * 70)
    for dim in [8, 32, 128, 512]:
        total = np.zeros(4)
        for t in range(REPEATS):
            scores = make_scores(dim, 1000 * t + dim)
            p_plain = softmax_row(scores)
            p_scaled = softmax_row(scores / np.sqrt(dim))
            total += [p_plain.max(), sensitivity(p_plain),
                      p_scaled.max(), sensitivity(p_scaled)]
        total /= REPEATS
        print(f"  {dim:>4}  |  {total[0]:>16.4f}{total[1]:>12.4f}  |  "
              f"{total[2]:>16.4f}{total[3]:>12.4f}")

    # ---------------- 一个具体的例子 ----------------
    print("\n" + "=" * 74)
    print("d = 512 时'苹果'那一行，长什么样")
    print("=" * 74)
    scores = make_scores(512, 512)

    for name, row_scores in [("不缩放", scores), ("除以 √d", scores / np.sqrt(512))]:
        weights = softmax_row(row_scores)
        print(f"\n  {name}：")
        order = sorted(range(len(SENTENCE)), key=lambda j: -weights[j])
        for j in order:
            mark = "  <- 只有它还站着" if (name == "不缩放" and j == order[0]) else ""
            print(f"    {pad(SENTENCE[j], 6)}{weights[j]:.8f}{mark}")
        print(f"    最大权重 = {weights.max():.8f}   "
              f"有效词数 = {effective_count(weights):.2f}   "
              f"梯度 = {sensitivity(weights):.10f}")

    print("\n  不缩放的那一行，权重被压成了 1 和十个 0——softmax 在这里")
    print("  已经退化成'只挑一个词'，而且梯度是 0.0000000001，约等于没有。")
    print("  除以 √d 之后，权重还是一组能看清轻重的数，梯度也还在。")

    print("\n  所以 √d 不是装饰：它把分数的尺度钉在 1 附近，")
    print("  让 softmax 既不会摊平成一锅粥，也不会塌成一个点。")


if __name__ == "__main__":
    main()
