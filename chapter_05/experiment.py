"""
这个文件摆出三把尺子，让它们量同样的几张权重表，看谁说得清楚。

    尺子一：第 4 章的"答错了几句"——一个 0 到 10 的整数
    尺子二：平方误差——把"分数差"当预测值，离目标差多少就平方一下
    尺子三：交叉熵（本章的新东西）——正确答案的概率越低，数越大

然后看概率是怎么从分数变出来的：sigmoid 和 softmax。

最后对比三张表：全 0 的表、第 4 章训出来的表、本章用损失训出来的表。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 别的章节也可能有同名的 before.py / after.py，先清掉缓存
sys.modules.pop("before", None)
sys.modules.pop("after", None)

from after import (
    ADJUST_AMOUNT,
    CATEGORIES,
    CORPUS,
    VOCABULARY,
    WEIGHT_TABLE,
    adjust_weights_once,
    average_loss,
    pad,
    softmax,
    split_sentence,
)


# ---------- 打分、判断、找错句：三个尺子都要用的公共部分 ----------

def score_with(words, table):
    scores = [0.0, 0.0]
    for word in words:
        if word not in table:
            continue
        scores[0] += table[word][0]
        scores[1] += table[word][1]
    return scores


def predict_with(words, table):
    scores = score_with(words, table)
    return CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]


def count_wrong(table):
    """尺子一：答错了几句。"""
    wrong = 0
    for sentence, label in CORPUS:
        if predict_with(split_sentence(sentence), table) != label:
            wrong += 1
    return wrong


def signed_error(words, label, table):
    """尺子二：平方误差。把分数差当预测值，目标取 +1（科技）或 -1（食品）。"""
    scores = score_with(words, table)
    difference = scores[0] - scores[1]
    target = 1.0 if label == "科技" else -1.0
    return (difference - target) ** 2


def average_squared_error(table):
    total = 0.0
    for sentence, label in CORPUS:
        total += signed_error(split_sentence(sentence), label, table)
    return total / len(CORPUS)


def loss_with(words, label, table):
    """尺子三：交叉熵。用临时替换过的表算，方便比较不同的表。"""
    scores = score_with(words, table)
    probabilities = softmax(scores)
    return -math.log(probabilities[CATEGORIES.index(label)])


def average_loss_with(table):
    total = 0.0
    for sentence, label in CORPUS:
        total += loss_with(split_sentence(sentence), label, table)
    return total / len(CORPUS)


# ---------- 用第 4 章的规则训一张表出来（和上一章一模一样） ----------

def train_by_mistakes(corpus):
    table = {word: [0.0, 0.0] for word in VOCABULARY}
    for _ in range(20):
        wrong_count = 0
        for sentence, label in corpus:
            words = split_sentence(sentence)
            answer = predict_with(words, table)
            if answer == label:
                continue
            wrong_count += 1
            correct_index = CATEGORIES.index(label)
            wrong_index = CATEGORIES.index(answer)
            for word in words:
                if word not in table:
                    continue
                table[word][correct_index] += 1.0
                table[word][wrong_index] -= 1.0
        if wrong_count == 0:
            break
    return table


def copy_table(table):
    return {word: list(weights) for word, weights in table.items()}


def format_loss(value):
    """损失会越变越小，小到小数点后四位看不出来，所以很小的数改用科学计数法。"""
    if value < 0.0001:
        return f"{abs(value):.2e}"
    return f"{value:.4f}"


def main():
    mistake_table = train_by_mistakes(CORPUS)

    # 表 A：上一章训出来的表
    table_a = copy_table(mistake_table)
    # 表 B：只把"很"动一点点，让「苹果芯片很强」刚好答错
    table_b = copy_table(mistake_table)
    table_b["很"] = [-1.5, 1.5]
    # 表 C：把"很"改得离谱，让「苹果芯片很强」错得很惨
    table_c = copy_table(mistake_table)
    table_c["很"] = [-20.0, 20.0]

    print("=" * 78)
    print("实验一：三把尺子，量三张表")
    print("=" * 78)
    print("  表 A：第 4 章的规则训出来的表")
    print("  表 B：在表 A 上，把'很'从 [0.0, 0.0] 改成 [-1.5, 1.5]")
    print("  表 C：在表 A 上，把'很'从 [0.0, 0.0] 改成 [-20.0, 20.0]")
    print("  （'很'只在「苹果芯片很强」这一句科技句里出现过，所以这两处改动")
    print("    都只影响这一句：B 让它差一点点答错，C 让它错得离谱。）")
    print()
    print(f"  {'':<8}{'答错几句':>10}{'平方误差':>12}{'交叉熵':>14}")
    for name, table in [("表 A", table_a), ("表 B", table_b), ("表 C", table_c)]:
        print(f"  {pad(name, 8)}{count_wrong(table):>10}"
              f"{average_squared_error(table):>12.2f}"
              f"{format_loss(average_loss_with(table)):>14}")
    print()

    print("  三张表里，「苹果芯片很强」这句的分数差是这样的：")
    for name, table in [("表 A", table_a), ("表 B", table_b), ("表 C", table_c)]:
        scores = score_with(split_sentence("苹果芯片很强"), table)
        print(f"    {pad(name, 8)} 科技 {scores[0]:>6.1f} / 食品 {scores[1]:>6.1f}"
              f"   差 {scores[0] - scores[1]:>6.1f}   ->  {predict_with(split_sentence('苹果芯片很强'), table)}")
    print()
    print("  读这张表：")
    print("    尺子一（答错几句）：表 B 和表 C 都是 1，一模一样。")
    print("                       可 B 是「差一点点」，C 是「错得离谱」，它分不出来。")
    print("    尺子二（平方误差）：分得出 B 和 C，但它给的数很难解释——")
    print("                       它想让分数差正好等于 1，比 1 大也要罚。")
    print(f"    尺子三（交叉熵）：  B 和 C 差了将近 "
          f"{average_loss_with(table_c) / average_loss_with(table_b):.0f} 倍。")
    print("                       错得越离谱，罚得越狠——这正是我们要的。")
    print()

    print("=" * 78)
    print("实验二：平方误差到底哪里不对")
    print("=" * 78)
    print("  挑表 A 里的四句话，看两个尺子各给多少：")
    print()
    print(f"  {'句子':<18}{'赢了几分':>8}{'平方误差':>10}{'交叉熵':>10}   说明")
    notes = {
        "苹果芯片很强": "只赢一点点，最勉强",
        "苹果发布新手机": "赢 4 分，还行",
        "苹果发布新芯片": "赢 6 分，挺稳",
        "这个苹果真甜": "赢 8 分，最有把握",
    }
    for sentence in ["苹果芯片很强", "苹果发布新手机", "苹果发布新芯片", "这个苹果真甜"]:
        label = dict(CORPUS)[sentence]
        words = split_sentence(sentence)
        scores = score_with(words, table_a)
        margin = abs(scores[0] - scores[1])
        print(f"  {pad(sentence, 18)}{margin:>8.1f}"
              f"{signed_error(words, label, table_a):>10.2f}"
              f"{loss_with(words, label, table_a):>10.3f}   {notes[sentence]}")
    print()
    print("  看最上面和最下面两行：")
    print("    '苹果芯片很强'最勉强（只赢 2 分），平方误差给它 1.00，说它最准；")
    print("    '这个苹果真甜'最有把握（赢 8 分），平方误差给它 49.00，说它错得最狠。")
    print("  它把话说反了。")
    print()
    print("  原因：平方误差想让'分数差'正好等于 1。")
    print("  于是越过 1 的部分全变成了'错误'——越有把握，错得越多。")
    print("  交叉熵不这样：它只关心'正确答案的概率离 1 有多远'，越接近 1 越好。")
    print()

    print("=" * 78)
    print("实验三：概率是怎么从分数变出来的")
    print("=" * 78)
    print("  一个类别的时候，用这个式子把分数变成概率：")
    print("     概率 = 1 / (1 + e 的 (-分数) 次方)")
    print()
    for value in [-4, -2, -1, 0, 1, 2, 4]:
        probability = 1.0 / (1.0 + math.exp(-value))
        print(f"    分数 {value:>3}  ->  概率 {probability:.4f}")
    print()
    print("  分数是 0 的时候概率正好 0.5——两个选项各占一半。")
    print("  分数越大越接近 1，越小越接近 0。")
    print()
    print("  两个类别的时候，我们让两个概率'抢'，加起来正好等于 1：")
    for scores in [[1.7, 1.2], [1.0, -1.0], [-0.1, 0.1], [8.0, 0.0]]:
        probabilities = softmax(scores)
        print(f"    分数 {str(scores):<14} ->  概率 "
              f"[{probabilities[0]:.4f}, {probabilities[1]:.4f}]")
    print()
    print("  顺便验证一个巧合：分数是 [d, 0] 的时候，softmax 算出来的第一个概率，")
    print("  和上面那个式子算出来的完全一样：")
    for d in [-2.0, 0.0, 1.0]:
        first = softmax([d, 0.0])[0]
        single = 1.0 / (1.0 + math.exp(-d))
        print(f"    d = {d:>4}:  softmax {first:.6f}   那个式子 {single:.6f}")
    print()

    print("=" * 78)
    print("实验四：三张表，两个尺子")
    print("=" * 78)
    empty_table = {word: [0.0, 0.0] for word in VOCABULARY}

    # 用 after.py 里的办法，从全 0 出发，一路用损失这把尺子往下改
    for _ in range(30):
        _, changed = adjust_weights_once(CORPUS, amount=ADJUST_AMOUNT)
        if changed == 0:
            break
    loss_table = copy_table(WEIGHT_TABLE)

    print(f"  {'表':<24}{'正确率':>8}{'交叉熵':>12}")
    for name, table in [
        ("全 0 的表（起点）", empty_table),
        ("第 4 章的规则训出来的", mistake_table),
        ("本章用损失训出来的", loss_table),
    ]:
        right = 10 - count_wrong(table)
        print(f"  {pad(name, 24)}{right:>6}/10{format_loss(average_loss_with(table)):>12}")
    print()
    print("-" * 78)
    print("  第 4 章的尺子会说：后两张都是 10/10，一样好。")
    print("  这一章的尺子说：")
    for name, table in [("全 0 的表", empty_table),
                        ("第 4 章的表", mistake_table),
                        ("本章的表", loss_table)]:
        print(f"    {pad(name, 14)} {format_loss(average_loss_with(table))}")
    print("  差了上千倍。正确率到 10/10 就到头了，损失不会。")
    print()

    print("=" * 78)
    print("实验五：有了损失这把尺子，第 4 章那个问题终于能回答了")
    print("=" * 78)
    print("  " + "每次改多少" + " 会怎么样？（每次都从全 0 的表出发）")
    print()
    print(f"  {'每次改多少':<12}{'轮数':>6}{'最终损失':>14}{'正确率':>8}")
    for amount in [0.05, 0.5, 2.0, 5.0]:
        for word in VOCABULARY:
            WEIGHT_TABLE[word][:] = [0.0, 0.0]
        rounds = 0
        for _ in range(300):
            _, changed = adjust_weights_once(CORPUS, amount=amount)
            rounds += 1
            if changed == 0:
                break
        print(f"  {amount:<12}{rounds:>6}{format_loss(average_loss(CORPUS)):>14}"
              f"{10 - count_wrong(WEIGHT_TABLE):>6}/10")
    print()
    print("  第 4 章我们试过这件事，那时候四个步长的训练轨迹一模一样——")
    print("  因为当时唯一的尺子是'对/错'，它连'改得好不好'都看不出来。")
    print("  现在不一样了：改小一点要走几百轮，改大一点几轮就到了。")
    print("  尺子变细了，'每次改多少'才第一次成为一个真的问题。")


if __name__ == "__main__":
    main()
