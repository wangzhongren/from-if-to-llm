"""
这个文件回答一个问题：每次改多少？

after.py 里我们把"每次改多少"定成了 1.0。这个数是我们随手定的。
这个文件拿四个不同的数各训一遍，看看差别。

然后我们会发现一件更麻烦的事：几个不同的权重表都能全对，
而"哪个更好"这个问题，我们手上居然没有东西可以回答。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 别的章节也可能有同名的 before.py / after.py，先清掉缓存
sys.modules.pop("before", None)
sys.modules.pop("after", None)

from before import HAND_FILLED_TABLE
from after import CATEGORIES, CORPUS, VOCABULARY, pad, split_sentence

ROUNDS = 10
STEP_CHOICES = [0.01, 0.1, 1.0, 10.0]


def make_empty_table():
    return {word: [0.0, 0.0] for word in VOCABULARY}


def predict(words, table):
    scores = [0.0, 0.0]
    for word in words:
        if word not in table:
            continue
        scores[0] += table[word][0]
        scores[1] += table[word][1]
    return CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]


def train_one_round(corpus, table, step):
    wrong_count = 0
    for sentence, label in corpus:
        words = split_sentence(sentence)
        answer = predict(words, table)
        if answer == label:
            continue
        wrong_count += 1
        correct_index = CATEGORIES.index(label)
        wrong_index = CATEGORIES.index(answer)
        for word in words:
            if word not in table:
                continue
            table[word][correct_index] += step
            table[word][wrong_index] -= step
    return wrong_count


def train(corpus, step, rounds):
    """从头训一遍，返回 (每轮答错数, 训完的表)。"""
    table = make_empty_table()
    history = []
    for _ in range(rounds):
        history.append(train_one_round(corpus, table, step))
    return history, table


def measure(corpus, table):
    right = 0
    for sentence, label in corpus:
        if predict(split_sentence(sentence), table) == label:
            right += 1
    return right


def biggest_weight(table):
    return max(abs(value) for weights in table.values() for value in weights)


def main():
    print("=" * 78)
    print("实验一：把'每次改多少'换成四个不同的数，各训练 10 轮")
    print("=" * 78)
    print("  每次改多少    每轮答错的句数                                        最终正确率")
    tables = {}
    for step in STEP_CHOICES:
        history, table = train(CORPUS, step=step, rounds=ROUNDS)
        tables[step] = table
        numbers = " ".join(f"{count:>2}" for count in history)
        print(f"  {step:<12}  {numbers}      {measure(CORPUS, table):>2}/10")
    print()
    print("  四行一模一样。")
    print()
    print("  我们本来以为会看到'改小了学得慢、改大了来回跳'。结果这四个数")
    print("  给出的训练轨迹完全相同。原因不难想：")
    print()
    print("    我们判断类别，靠的是两个分数谁大。")
    print("    把所有权重同时乘以 100，两个分数也同时乘以 100，谁大谁小没变。")
    print("    所以'每次改多少'只改数字的大小，不改对错。")
    print()
    print("  换句话说：在第 4 章的规则下，'每次改多少'是一个**不影响对错**的数。")
    print("  那它影响了什么？只有权重的大小。见实验二。")
    print()

    print("=" * 78)
    print("实验二：几个训好的表都能全对，它们一样吗？")
    print("=" * 78)
    print(f"  {'每次改多少':<12} {'正确率':<8} {'权重最大的绝对值':<16}")
    for step in STEP_CHOICES:
        table = tables[step]
        print(f"  {step:<12} {measure(CORPUS, table):>2}/10    {biggest_weight(table):.2f}")
    print()
    print("  再看两个表的细节（挑 4 个词）：")
    print(f"  {'词':<8} {'改 1.0 训出来的':<20} {'改 10.0 训出来的':<20}")
    for word in ["苹果", "芯片", "很", "好吃"]:
        small = tables[1.0][word]
        big = tables[10.0][word]
        print(f"  {pad(word, 8)} 科技 {small[0]:>6.1f} 食品 {small[1]:>6.1f}     "
              f"科技 {big[0]:>6.1f} 食品 {big[1]:>6.1f}")
    print()
    print("  同样是 10/10，一个表的数字是个位数，另一个是两位数。")
    print()

    print("=" * 78)
    print("实验三：训练出来的表，和手填的表，比一比")
    print("=" * 78)
    print("  正确率上它们没有区别，都是 10/10。")
    print("  挑同一句话「苹果芯片很强」，看它在三张表里的处境：")
    print()
    tables_to_show = [
        ("第 3 章手填的表", HAND_FILLED_TABLE),
        ("训练出来的表（每次改 1.0）", tables[1.0]),
        ("训练出来的表（每次改 10.0）", tables[10.0]),
    ]
    for name, table in tables_to_show:
        words = split_sentence("苹果芯片很强")
        scores = [0.0, 0.0]
        for word in words:
            if word not in table:
                continue
            scores[0] += table[word][0]
            scores[1] += table[word][1]
        print(f"  {pad(name, 30)} 科技 {scores[0]:>6.1f} / 食品 {scores[1]:>6.1f}")
    print()
    print("  三张表都答对了这一句。可它们赢的幅度是 0.5、2.0、20.0。")
    print("  如果我问你'哪一张更有把握'，你打算怎么回答？")
    print()
    print("  看正确率？都是 10/10，分不出来。")
    print("  看某一个权重？手填的表里'芯片'是 1.5，训出来的是 2.0 或 20.0，")
    print("  这些数之间没有可比性——它们只是被'每次改多少'整体放大过。")
    print()
    print("-" * 78)
    print("第 3 章我们靠'对了几句'来判断一张表好不好。")
    print("可一旦几张表都是 10/10，这个指标就再也说不出话了。")
    print("我们需要的是一把更细的尺子——它能说出'错得有多离谱'，")
    print("而不只是'错了还是对了'。")


if __name__ == "__main__":
    main()
