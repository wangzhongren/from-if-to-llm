"""这一章最重要的实验。

我们要亲眼看到三件事：

1. 损失随一个参数变化的样子 —— 往哪边挪，它会变小。
2. 数值法算出来的梯度和手推公式算出来的梯度，是同一个东西。
3. 学习率选得对不对，结果差多少。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from after import (  # noqa: E402
    CATEGORIES,
    CORPUS,
    VOCABULARY,
    WEIGHT_TABLE,
    accuracy,
    average_loss,
    gradient,
    numeric_gradient,
    reset_table,
    score,
    softmax,
    split_sentence,
    train,
)

# 数值法每算一个梯度，要跑这么多次完整的前向计算：
# 32 个参数 × 正负各挪一次
FORWARDS_PER_NUMERIC_GRADIENT = len(VOCABULARY) * 2 * 2


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


def loss_with_floor(corpus):
    """给极端的学习率准备的版本。

    参数一旦飞出去，正确类别的概率会下溢到 0，-log(0) 直接报错。
    这里给它夹一个极小的下界，保证学习率那一张表能完整打印出来。
    """
    total = 0.0
    for sentence, label in corpus:
        probability = softmax(score(split_sentence(sentence)))[
            CATEGORIES.index(label)
        ]
        total += -math.log(max(probability, 1e-300))
    return total / len(corpus)


# ------------------------------------------------------------------ 第 1 步


def step1_watch_loss():
    print("=" * 66)
    print("第 1 步：损失到底长什么样")
    print("=" * 66)
    print("所有参数都是 0 的时候，两个类别的概率各 0.5，损失 = ln 2 = 0.6931。")
    print("现在我们只动一个参数 ——「发布」在科技这一类里的权重 ——")
    print("把它分别设成几个值，看看损失怎么变。")
    print()

    reset_table()
    base = average_loss(CORPUS)
    print(f"  {'「发布」的科技权重':>22}{'损失':>12}{'比 0 的时候':>14}")
    for value in (-0.05, -0.01, 0.0, 0.01, 0.05):
        WEIGHT_TABLE["发布"][0] = value
        current = average_loss(CORPUS)
        print(f"  {value:>22.2f}{current:>12.6f}{current - base:>+14.6f}")
    reset_table()
    print()
    print("  4 句科技句里有「发布」。把这个参数往正方向挪，")
    print("  科技类的分数变高，损失就变小 —— 这就是我们要的那个方向。")
    print()


# ------------------------------------------------------------------ 第 2 步


def step2_numeric_gradient():
    print("=" * 66)
    print("第 2 步：把 32 个方向全算出来（数值梯度）")
    print("=" * 66)
    print("办法：把某个参数往正负两边各挪 h=0.001，然后")
    print("      (损失(参数 + h) - 损失(参数 - h)) / (2h)")
    print("  它衡量的是「这个参数每往正方向走一点，损失会涨多快」。")
    print("  正数 = 往正方向走损失会变大，所以要往负方向改。")
    print()

    reset_table()
    grads = numeric_gradient(WEIGHT_TABLE, CORPUS)

    print(f"  {pad('词', 8)}{'科技类的梯度':>14}{'食品类的梯度':>14}"
          f"   出现在几句科技句 / 食品句")
    for word in VOCABULARY:
        in_tech = sum(1 for sentence, label in CORPUS
                      if label == "科技" and word in split_sentence(sentence))
        in_food = sum(1 for sentence, label in CORPUS
                      if label == "食品" and word in split_sentence(sentence))
        print(f"  {pad(word, 8)}{grads[word][0]:>14.5f}{grads[word][1]:>14.5f}"
              f"        {in_tech} / {in_food}")
    print()
    print("  几个可以直接读出来的结论：")
    print("    「发布」的科技类梯度是负的 —— 该往正方向加。")
    print("    「好吃」「甜」的科技类梯度是正的 —— 该往负方向减。")
    print("    梯度的大小，和这个词在两类句子里各出现过几次有关。")
    print()


# ------------------------------------------------------------------ 第 3 步


def step3_check_direction():
    print("=" * 66)
    print("第 3 步：梯度算出来的方向，和一次一次试出来的方向对得上吗")
    print("=" * 66)

    reset_table()
    grads = numeric_gradient(WEIGHT_TABLE, CORPUS)
    h = 0.001
    base = average_loss(CORPUS)

    WEIGHT_TABLE["发布"][0] = h
    right = average_loss(CORPUS)
    WEIGHT_TABLE["发布"][0] = -h
    left = average_loss(CORPUS)
    reset_table()

    print(f"  把「发布」的科技权重往右挪 {h}：损失 {base:.6f} -> {right:.6f}")
    print(f"  把「发布」的科技权重往左挪 {h}：损失 {base:.6f} -> {left:.6f}")
    print()
    print("  往右挪损失变小了 —— 所以要往右改，也就是给这个参数加一点点。")
    print(f"  数值法给出的梯度 = {grads['发布'][0]:+.5f}")
    print("  更新公式 w = w - 学习率 × 梯度 里的减号，")
    print("  配合一个负的梯度，正好就是往右改。")
    print("  这就是为什么公式里是减号，不是加号。")
    print()


# ------------------------------------------------------------------ 第 4 步


def step4_formula_matches():
    print("=" * 66)
    print("第 4 步：数值法 vs 手推公式")
    print("=" * 66)
    print(f"  数值法很慢：算一次梯度要跑 {FORWARDS_PER_NUMERIC_GRADIENT} 次前向。")
    print("  所以我们对这个模型手推了一个公式：(预测概率 - 正确答案)。")
    print("  公式对不对？用数值法验一遍就知道了。")
    print()

    reset_table()
    # 先随便走几步，让参数离 0 远一点，避免在特殊点上验证
    grads = gradient(WEIGHT_TABLE, CORPUS)
    for word in VOCABULARY:
        for category_index in (0, 1):
            WEIGHT_TABLE[word][category_index] -= 0.5 * grads[word][category_index]

    numeric = numeric_gradient(WEIGHT_TABLE, CORPUS)
    formula = gradient(WEIGHT_TABLE, CORPUS)

    print(f"  {pad('词', 8)}{'数值法':>12}{'公式':>12}{'差':>12}")
    worst = 0.0
    for word in VOCABULARY:
        for category_index in (0, 1):
            difference = abs(numeric[word][category_index]
                             - formula[word][category_index])
            worst = max(worst, difference)
            if category_index == 0:
                print(f"  {pad(word, 8)}{numeric[word][0]:>12.6f}"
                      f"{formula[word][0]:>12.6f}{difference:>12.2e}")
    print()
    print(f"  32 个参数里，最大的差是 {worst:.2e}。")
    print("  这不是巧合，是同一件事的两种算法。")
    print()


# ------------------------------------------------------------------ 第 5 步


def step5_same_budget():
    print("=" * 66)
    print("第 5 步：花掉同样多的计算，两种方法各走了多远")
    print("=" * 66)
    print("  一次「前向」= 把 10 句话完整算一遍损失。")
    print(f"  数值法每走一步要挪 32 个参数 × 正负各一次 = "
          f"{FORWARDS_PER_NUMERIC_GRADIENT} 次前向。")
    print("  公式法每走一步只要 2 次前向（一次算梯度，一次记损失）。")
    print()

    # 数值法走 30 步
    reset_table()
    numeric_steps = 30
    lr = 0.5
    for _ in range(numeric_steps):
        grads = numeric_gradient(WEIGHT_TABLE, CORPUS)
        for word in VOCABULARY:
            for category_index in (0, 1):
                WEIGHT_TABLE[word][category_index] -= (
                    lr * grads[word][category_index]
                )
    numeric_loss = average_loss(CORPUS)
    numeric_forwards = numeric_steps * FORWARDS_PER_NUMERIC_GRADIENT

    # 公式法用掉同样多的前向次数
    reset_table()
    formula_steps = numeric_forwards // 2
    history = train(steps=formula_steps, lr=lr)

    print(f"  {'方法':<10}{'步数':>6}{'前向次数':>10}"
          f"{'最后的损失':>14}{'正确率':>8}")
    print(f"  {'数值法':<10}{numeric_steps:>6}{numeric_forwards:>10}"
          f"{numeric_loss:>14.6f}{accuracy(CORPUS):>8}/10")
    print(f"  {'公式法':<10}{formula_steps:>6}{formula_steps * 2:>10}"
          f"{history[-1]:>14.8f}{accuracy(CORPUS):>8}/10")
    print()
    print(f"  同样花掉 {numeric_forwards} 次前向：数值法把损失降到 {numeric_loss:.4f}，")
    print(f"  公式法降到了 {history[-1]:.4f} —— 差了 {numeric_loss / history[-1]:.6f} 倍。")
    print("  而且这个差距只会越拉越大。")
    print("  注意：两条路用的更新规则一模一样，都是 w = w - 学习率 × 梯度，")
    print("  差别只在梯度怎么算出来。")
    print()


# ------------------------------------------------------------------ 第 6 步


def step6_learning_rate():
    print("=" * 66)
    print("第 6 步：学习率选多大")
    print("=" * 66)
    print("  梯度只告诉我们方向，学习率告诉我们这一步走多远。")
    print()

    def train_quietly(learning_rate, steps=200):
        reset_table()
        for _ in range(steps):
            grads = gradient(WEIGHT_TABLE, CORPUS)
            for word in VOCABULARY:
                for category_index in (0, 1):
                    WEIGHT_TABLE[word][category_index] -= (
                        learning_rate * grads[word][category_index]
                    )

    print(f"  {'学习率':>10}{'200 步后的损失':>18}{'正确率':>10}"
          f"{'参数最大绝对值':>16}")
    for learning_rate in (0.0005, 0.05, 0.5, 5.0, 20.0, 1000.0):
        train_quietly(learning_rate)
        biggest = max(abs(value) for word in VOCABULARY
                      for value in WEIGHT_TABLE[word])
        print(f"  {learning_rate:>10}{loss_with_floor(CORPUS):>18.6f}"
              f"{accuracy(CORPUS):>10}/10{biggest:>16.2f}")
    reset_table()
    print()
    print("  0.0005：太胆小。200 步走完了，损失几乎没动，还有 1 句分错。")
    print("  0.5   ：我们选它。损失降到 0.018，10 句话全对。")
    print("  5.0 / 20.0：损失更小，但参数被推得越来越大。")
    print("  1000.0：这一行的 0.000000 不是「学得完美」。概率被推到了 1.0，")
    print("         -log(1) 正好是 0 —— 尺子到顶了，量不出差别。")
    print("         看最后一列：参数已经涨到 200。")
    print()
    print("  为什么学习率大不会炸掉？因为这 10 句话本来就是能分开的：")
    print("  参数越大，模型越有把握，损失越小，它根本没有一个「最低点」可以停。")
    print("  损失一路降下去，并不总是好消息。")
    print()
    print("  学习率该填多少，到今天也没有公式，只能试。")


def main():
    step1_watch_loss()
    step2_numeric_gradient()
    step3_check_direction()
    step4_formula_matches()
    step5_same_budget()
    step6_learning_rate()
    print()
    print("=" * 66)
    print("总结")
    print("=" * 66)
    print("  1. 损失是一个关于 32 个参数的函数，往某个方向挪，它会变小。")
    print("  2. 「往哪边挪、有多陡」这张表，就是梯度。数值法能算，公式也能算。")
    print("  3. w = w - 学习率 × 梯度，配合固定的学习率，就能把损失压到 0。")
    print()
    print("  这就是机器学习的训练。到这里，我们已经把它完整实现了。")
    print("  但是请注意第 4 步里那个公式 —— 它是我们针对这一个模型手推的。")
    print("  换成两层网络呢？那个公式要重新推一遍。")


if __name__ == "__main__":
    main()
