"""第 23 章 before：第 22 章的模型 + 每次都挑最大的那一个。

上一章我们让模型写出了诗，用的循环是这样的：

    把已经写好的字喂给模型
    拿它给出的概率最大的那个字，接到后面
    重复

这个做法有个名字，叫**贪心**（greedy）。它看起来是「最保险」的选择：
每一步都选最可能的那一个，整句话不就应该是最可能的吗？

这个文件只做一件事：**把贪心的毛病摆到台面上。**

毛病有三个，跑一遍就能看到：

  1. 同一个开头，跑多少次结果都一样。它没有「选择」这件事。
  2. 开头一旦离开语料，它接出来的东西很快就崩。
  3. 它写出来的永远是那 38 个字的循环——它在复读，不是在写。

第 3 条不能全怪贪心。第 1、2 条才是。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import cross_entropy, no_grad

from after import (  # noqa: E402
    CONTEXT,
    CORPUS,
    CharTokenizer,
    build_batches,
    train_default_model,
)


def greedy_continue(model, tokenizer, prompt, length):
    """贪心：每一步都挑概率最大的那一个。

    注意这个函数里**一个随机数都没有**。
    """
    ids = list(tokenizer.encode(prompt))
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-CONTEXT:])[None, :]
            logits = model(window)
            ids.append(int(logits.data[0, -1].argmax()))
    return tokenizer.decode(ids)


def main():
    tokenizer = CharTokenizer(CORPUS)
    print("=" * 60)
    print("准备")
    print("=" * 60)
    print("  模型和第 22 章完全一样：同一个结构、同一个种子、同一段语料。")
    model, inputs, targets = train_default_model(tokenizer)
    with no_grad():
        loss = cross_entropy(model(inputs), targets).item()
    print(f"  参数量 {model.parameter_count():,}   训练集 loss {loss:.4f}")

    print()
    print("=" * 60)
    print("毛病一：同一个开头，跑多少次都一样")
    print("=" * 60)
    prompt = "举头望明月低头"
    print(f"  开头：{prompt}")
    print()
    for run_index in range(3):
        print(f"    第 {run_index + 1} 次：{greedy_continue(model, tokenizer, prompt, 20)}")
    print()
    print("  三次完全一样。这不是巧合——`greedy_continue` 里一个随机数都没有。")
    print("  给定同一个开头，它是**算出来的**，每次都一样。")

    print()
    print("=" * 60)
    print("毛病二：开头一离开语料，很快就崩")
    print("=" * 60)
    print("  下面三个开头用的都是词表里的字，但语料里从来不会连着出现：")
    print()
    for prompt in ["故乡故乡故乡故乡故乡故乡故乡故乡",
                   "月月月月月月月月月月月月月月月月",
                   "水水水水水水水水水水水水水水水水"]:
        text = greedy_continue(model, tokenizer, prompt, 26)
        print(f"    {prompt[:8]}...")
        print(f"      -> {text}")
        print()
    print("  它一开始还能顺着走，走几步就散了。")
    print("  因为贪心每一步都只看「现在最像什么」，走错了也回不了头。")
    print("  它没有「退一步再看」这种选项——错一步，后面就全在错的基础上往前推。")

    print()
    print("=" * 60)
    print("毛病三：它写出来的永远是那 38 个字")
    print("=" * 60)
    text = greedy_continue(model, tokenizer, "床前", 120)
    print("  接 120 个字：")
    for start in range(0, min(len(text), 120), 20):
        print(f"    {text[start:start + 20]}")
    print()
    print("  两首诗来回循环。说到底，语料总共只有这 38 个字，")
    print("  而贪心每一步都在走「最确定」的那条路——")
    print("  最确定的那条路，就是语料本身。")
    print()
    print("  这一条不能全怪贪心。但前两条是它自己的毛病。")

    print()
    print("=" * 60)
    print("那么")
    print("=" * 60)
    print("  模型在每个位置给出的不是一个字符，而是 33 个概率。")
    print("  我们刚才做的事情是：把最大的那个拿走，剩下的 32 个全扔了。")
    print()
    print("  被扔掉的 32 个概率里，装着什么？")
    print("  如果我们在里面抽一次呢——")
    print("  同一个开头，会不会写出不一样的东西？")


if __name__ == "__main__":
    main()
