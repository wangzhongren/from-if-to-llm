"""第 15 章 experiment.py —— 一个头 vs 四个头，做同一个任务。

对照实验：同一个模型、同一批数据、同样的参数总量，只改"切几个头"。

  1 个头：每个头的维度是 24，一共 1 套 Q/K/V
  4 个头：每个头的维度是 6，一共 4 套 Q/K/V

两边的参数数量几乎一样。跑完以后看：

  - 损失曲线（两边都能把任务做对）
  - "他"这个位置拿到的权重：1 个头只有一行，4 个头有四行
  - 每一行有多"尖"：一行只能给一个偏好，四行可以各给各的

跑法：

    ./.venv/bin/python chapter_15/experiment.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    SENTENCE_2,
    WORD_ID,
    multi_head_attention,
    pad,
    train,
)

STEPS = 400


def he_row(params):
    """训练完之后，"他"那个位置每个头拿到的权重。"""
    ids = [WORD_ID[word] for word in SENTENCE_2]
    _, weights = multi_head_attention(params["table"][ids], params)
    return weights.data[:, SENTENCE_2.index("他"), :]


def show_row(title, rows, words):
    print(f"\n{title}")
    print("-" * 60)
    for head in range(rows.shape[0]):
        row = rows[head]
        top = sorted(range(len(words)), key=lambda j: -row[j])[:3]
        best = "、".join(f"{words[j]}({row[j]:.3f})" for j in top)
        print(f"  头 {head}：最大值 {row.max():.3f}   "
              f"前三个：{best}")


def main():
    words = SENTENCE_2

    print("=" * 60)
    print("同一个任务：用上下文猜每个词是谁")
    print("句子：", " ".join(words))
    print("=" * 60)

    print(f"\n  两边都训练 {STEPS} 步，同样的学习率，同样的随机种子。")
    params_one, history_one = train(steps=STEPS, num_heads=1)
    params_four, history_four = train(steps=STEPS, num_heads=4)

    print(f"\n  {'步数':>6}{'1 个头':>14}{'4 个头':>14}")
    print("  " + "-" * 34)
    for step in [0, 50, 100, 200, 400]:
        print(f"  {step:>6}{history_one[step]:>14.4f}{history_four[step]:>14.4f}")
    print("\n  两边最后都把这两句话背下来了（数据只有两句，这是必然的）。")
    print("  差别不在损失，在于每个位置能拿出几套权重。")

    rows_one = he_row(params_one)
    rows_four = he_row(params_four)

    print("\n" + "=" * 60)
    print("'他'这个位置，两边各拿到了什么")
    print("=" * 60)
    print("\n【1 个头】只有一行：")
    row = rows_one[0]
    for j, word in enumerate(words):
        print(f"    {pad(word, 6)}{row[j]:.4f}  {'#' * int(round(row[j] * 40))}")

    show_row("【4 个头】四行，各看各的：", rows_four, words)

    print("\n" + "=" * 60)
    print("四行放在一起看")
    print("=" * 60)
    top_one = int(np.argmax(rows_one[0]))
    print(f"\n  1 个头：只有一行，{rows_one[0][top_one]:.3f} 压在"
          f"'{words[top_one]}'上，剩下 {1 - rows_one[0][top_one]:.3f} 留给 8 个词。")
    print("           它想再表达点别的（比如'他 = 小王'），"
          "就只能从这 "
          f"{rows_one[0][top_one]:.3f} 里往外抠。")
    print("\n  4 个头：每一行各自挑自己的词——")
    for head in range(rows_four.shape[0]):
        top = int(np.argmax(rows_four[head]))
        print(f"           头 {head} 最关注 {words[top]}（{rows_four[head][top]:.3f}）")

    print("\n  注意 4 个头并没有比 1 个头'更尖'——"
          "它们做的事情是：把一份注意力拆成 4 份，")
    print("  各挑各的词。1 个头押了'考试'就押不了'书'；"
          "4 个头里，头 1 押'书'，头 3 押'考试'和'给了'。")
    print("  同样的参数预算，能同时表达的东西多了几倍。")


if __name__ == "__main__":
    main()
