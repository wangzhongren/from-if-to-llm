"""第 23 章 experiment：温度一格一格往上拧，看模型什么时候开始"敢"。

after.py 给的是一张快照。这个文件把它变成一条连续的线，
再量几个我们嘴上说不清的东西：

  1. 温度从 0.2 拧到 8，**同一个开头**写出来的东西什么时候开始不一样？
  2. 每个温度下，模型真正用到了几个候选？（叫"有效候选数"）
  3. 写出来的东西和语料"像不像"？（把它和语料的最长公共子串量出来）
  4. top-k 的 k 从 1 加到 33，和只改温度有什么不一样？

第 2 条尤其值得看。它会告诉你一个我们在第 22 章就已经埋下的结论：
**这个模型太自信了**，所以温度要拧得很大才看得出来效果。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from after import (  # noqa: E402
    CONTEXT,
    CORPUS,
    CharTokenizer,
    generate,
    next_token_probabilities,
    train_default_model,
)

PROMPT = "举头望明月低头思"
PROMPT_HEAD = "床前"
LENGTH = 24


def effective_candidates(probabilities, temperature=1.0):
    """有效候选数：把温度作用到概率上之后，exp(熵) 是多少。

    如果模型只在一个候选上有概率，这个数是 1。
    如果 33 个候选平分，这个数是 33。
    """
    adjusted = probabilities ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()
    adjusted = adjusted[adjusted > 0]
    entropy = -(adjusted * np.log(adjusted)).sum()
    return float(np.exp(entropy))


def longest_common_substring(text, corpus):
    """最长公共子串的长度。

    数字越大，说明它写出来的东西和语料里的一段越像——也就是越像在背书。
    """
    best = 0
    for start in range(len(text)):
        length = 0
        while (start + length < len(text)
               and text[start:start + length + 1] in corpus):
            length += 1
        best = max(best, length)
    return best


def main():
    tokenizer = CharTokenizer(CORPUS)
    print("=" * 60)
    print("准备")
    print("=" * 60)
    model, inputs, targets = train_default_model(tokenizer)
    probabilities = next_token_probabilities(model, tokenizer, PROMPT)
    print(f"  模型和第 22、23 章完全一样。开头：{PROMPT}")
    print()
    top = np.argsort(-probabilities)[:5]
    print("  这个位置模型给出的概率：")
    for index in top:
        print(f"    '{tokenizer.id_to_char[int(index)]}'  {probabilities[index]:.4f}")
    print()
    print(f"  最大的那个占了 {probabilities.max():.4f}，")
    print(f"  第二大的只有 {sorted(probabilities)[-2]:.6f}。")

    print()
    print("=" * 60)
    print("温度一格一格往上拧")
    print("=" * 60)
    print("  同一个开关、同一个开头，只改温度。每个温度跑 6 次，")
    print("  看看 6 次里写出了几种不同的东西。")
    print()
    print(f"  {'温度':>5}  {'有效候选数':>10}  {'6 次里有几种':>12}  {'最长复读':>8}")
    print("  " + "-" * 44)
    for temperature in (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        texts = [generate(model, tokenizer, PROMPT, LENGTH, temperature=temperature, seed=seed)
                 for seed in range(6)]
        distinct = len(set(texts))
        repeat = max(longest_common_substring(text, CORPUS) for text in texts)
        print(f"  {temperature:>5}  {effective_candidates(probabilities, temperature):>10.2f}"
              f"  {distinct:>12}  {repeat:>8}")

    print()
    print("  「有效候选数」是 exp(熵)：1 表示它只认真考虑一个候选，")
    print("  33 表示它在 33 个候选之间完全瞎猜。")
    print()
    print("  「最长复读」是它写出来的东西和语料的最长公共子串的长度。")
    print("  数字越大，越说明它在背书。")

    print()
    print("=" * 60)
    print("读这张表")
    print("=" * 60)
    print("  温度 0.2、0.5：有效候选数一直是 1.00，六次跑出来完全一样。")
    print("  温度 1.0 也只跑出两种，而最长复读还是 32——")
    print("  就算换了，也还是整段整段地复读语料。")
    print()
    print("  在这个区间里，「温度」这个旋钮几乎是坏的——")
    print("  不是温度没用，是模型**太自信了**。")
    print()
    print("  温度 2 开始松动（六次里有六种，有效候选数 1.11），")
    print("  温度 3、4 最长复读掉到 24、12。")
    print("  也就是说，要让这个模型真的写出点不一样的东西，")
    print("  温度得拧到 3 以上，而那时候它写出来的已经不太像话了。")
    print()
    print("  为什么这么不灵敏？")
    print("  因为训练集上的 loss 只有 0.0254，模型几乎把语料背下来了。")
    print("  它给出的概率分布是尖的：上面那个开头，第一名 0.9998，第二名 0.0001。")
    print("  要让第二名有机会，就得把温度拧得很大。")
    print()
    print("  一句话：**模型有多自信，决定了温度要多拧才能起作用。**")
    print("  这件事的根源是语料太小，第 24 章会专门回来看。")

    print()
    print("=" * 60)
    print("top-k 和温度管的不是同一件事")
    print("=" * 60)
    print("  固定温度 8.0，只改 k：")
    print()
    print(f"  {'k':>4}  {'6 次里有几种':>12}  {'最长复读':>8}   一次的样子")
    print("  " + "-" * 58)
    for top_k in (1, 2, 3, 5, 10, 33):
        texts = [generate(model, tokenizer, PROMPT, LENGTH, temperature=8.0,
                          top_k=top_k, seed=seed) for seed in range(6)]
        print(f"  {top_k:>4}  {len(set(texts)):>12}  "
              f"{max(longest_common_substring(text, CORPUS) for text in texts):>8}"
              f"   {texts[0][len(PROMPT):len(PROMPT) + 18]}")

    print()
    print("  k=1 就是贪心。k=33 就是什么都不挡。")
    print()
    print("  温度的作用是**重新分配概率**——它把 33 个候选全部留着，")
    print("  只是让原来很小的那些变大一点。")
    print("  所以温度拧大之后，那些本来几乎不可能的字也会被抽中，")
    print("  有时候会抽到特别离谱的。")
    print()
    print("  top-k 的作用是**直接砍掉尾巴**——它先把不可能的那些候选删掉，")
    print("  剩下的还是按原本的比例分。")
    print("  所以它既给了多样性，又不会写出太怪的东西。")
    print()
    print("  这两个旋钮今天几乎所有语言模型都带着。")
    print("  你在网页版里看到的「温度」滑块，就是第一个旋钮。")

    print()
    print("=" * 60)
    print("最后换一个更短的开头")
    print("=" * 60)
    print(f"  开头：{PROMPT_HEAD}")
    print()
    for label, options in (
        ("贪心         ", dict(temperature=1.0, top_k=1)),
        ("温度 4       ", dict(temperature=4.0)),
        ("top-k 3 + 温度 4", dict(temperature=4.0, top_k=3)),
    ):
        print(f"  {label}")
        for seed in range(3):
            print(f"    第 {seed + 1} 次：{generate(model, tokenizer, PROMPT_HEAD, 30, seed=seed, **options)}")
        print()

    print("  同一个模型，同一个开头。")
    print("  第一种写法每次都一样，后两种每次都不一样。")
    print("  这就是「生成」和「预测」的区别。")


if __name__ == "__main__":
    main()
