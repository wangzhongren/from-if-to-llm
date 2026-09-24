"""第 16 章 experiment.py —— 打乱词序，输出到底变不变。

三个实验：

  一、置换实验。把『狗咬人』的词序打乱成『人咬狗』，
      看不加位置信息和加了位置信息，输出各是什么样。
  二、位置编码的相似度表：相同距离的两个位置，向量有多像。
  三、两种位置信息对比：加了之后，输入还是不是"原来那个词"。

跑法：

    ./.venv/bin/python chapter_16/experiment.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    DIM,
    EMBEDDINGS,
    SENTENCE_A,
    SENTENCE_B,
    attention,
    index_positions,
    pad,
    sinusoidal_positions,
)

LENGTH = 11


def sentence_vector(words, position_table):
    output, _ = attention(words, position_table)
    return output.data, output.data.sum(axis=0)


def main():
    print("=" * 60)
    print("『狗 咬 人』 vs 『人 咬 狗』：同一批词，换了个顺序")
    print("=" * 60)

    table = sinusoidal_positions(LENGTH, DIM)
    no_position = np.zeros((LENGTH, DIM))

    # ---------------- 实验一 ----------------
    print("\n" + "=" * 60)
    print("实验一：打乱词序，输出变了没有")
    print("=" * 60)

    for name, position_table in [("不加位置信息", no_position), ("加了位置编码", table)]:
        output_a, sum_a = sentence_vector(SENTENCE_A, position_table[: len(SENTENCE_A)])
        output_b, sum_b = sentence_vector(SENTENCE_B, position_table[: len(SENTENCE_B)])

        # 『人』在 狗咬人 里排第 2，在 人咬狗 里排第 0
        ren_in_a = SENTENCE_A.index("人")
        same_row = np.allclose(output_b[0], output_a[ren_in_a])
        distance = float(np.linalg.norm(sum_a - sum_b))
        print(f"\n  【{name}】")
        print(f"    『人』在两个句子里的输出一样吗：{same_row}")
        print(f"    两句话的句子向量距离：{distance:.4f}")

    print("\n  不加位置信息时，输出只是把同一堆向量换了个顺序，")
    print("  所以整句话的表示（所有位置加起来）一模一样，距离 = 0。")

    # ---------------- 实验二 ----------------
    print("\n" + "=" * 60)
    print("实验二：位置编码的相似度，只看距离")
    print("=" * 60)
    print(f"\n  {'位置对':>12}{'距离':>6}{'相似度':>10}   同一距离的其他位置对")
    print("  " + "-" * 56)
    for distance in [0, 1, 2, 3, 5]:
        pairs = [(i, i + distance) for i in range(0, 6)]
        pairs = pairs[:3]
        values = [float(table[i] @ table[j]) for i, j in pairs]
        first = f"{pairs[0][0]} 和 {pairs[0][1]}"
        others = "、".join(f"{i}和{j}({table[i] @ table[j]:.3f})" for i, j in pairs[1:])
        print(f"  {first:>12}{distance:>6}{values[0]:>10.3f}   {others}")

    print("\n  距离一样，相似度就一样：0和1、1和2、3和4 都是 3.535；")
    print("  0和2、2和4 都是 2.564。这就是位置编码的好处——")
    print("  它记的不是「我在第几位」，而是「我们隔多远」。")
    print("  （注意它不是越远越不像：距离 5 反而比距离 2 更像一点，")
    print("   因为不同频率的波会周期性重合。关键是同一个距离永远同一个值。）")

    # ---------------- 实验三 ----------------
    print("\n" + "=" * 60)
    print("实验三：加上位置信息之后，这个向量还像「原来那个词」吗")
    print("=" * 60)
    word_vector = np.array(EMBEDDINGS["狗"])
    word_norm = np.linalg.norm(word_vector)
    index_table = index_positions(LENGTH, DIM)

    def similarity_to_word(position_vector):
        mixed = word_vector + position_vector
        return float(mixed @ word_vector / (np.linalg.norm(mixed) * word_norm))

    long_index = index_positions(51, DIM)
    long_table = sinusoidal_positions(51, DIM)

    print(f"\n  {'位置':>6}{'编号向量长度':>14}{'位置编码长度':>14}"
          f"{'编号向量还像狗吗':>18}{'位置编码还像狗吗':>18}")
    print("  " + "-" * 68)
    for i in [0, 1, 2, 5, 10, 50]:
        index_vector = long_index[i] if i < len(long_index) else index_table[-1]
        print(f"  {i:>6}{np.linalg.norm(index_vector):>14.2f}"
              f"{np.linalg.norm(long_table[i]):>14.2f}"
              f"{similarity_to_word(index_vector):>18.3f}"
              f"{similarity_to_word(long_table[i]):>18.3f}")

    print("\n  编号向量的长度随着位置一路涨（位置 50 时是 141.42），"
          "词向量那点数值被淹没了；")
    print("  位置编码的长度永远是 2.00，不管第几位。")
    print("  所以用编号向量，位置越靠后那个词就越不像自己；")
    print("  用位置编码，词义和位置的分量是稳定的。")


if __name__ == "__main__":
    main()
