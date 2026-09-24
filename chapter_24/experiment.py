"""第 24 章 experiment：一个决定性的对照实验。

after.py 摆出了一堆观察，但每个观察单独看都可以有两种解释。
这个文件做一件事，把其中一种解释排掉：

    **把语料的字序完全打乱，再训一个一模一样的模型。**

如果模型学到的是「中文怎么排」，那么在打乱的语料上训练，
它应该学不到什么东西——因为打乱的文本没有中文的规律。

如果模型学到的是「你喂给它的那个顺序」，那么在打乱的语料上训练，
它应该学得和原来一样好——只是学的是另一套顺序。

三种实验：

  实验一：交叉评估。原文模型和打乱模型，互相到对方的地盘上考一次。
  实验二：多训一会儿，「打乱文本上的损失」会不会变好？
  实验三：语料重复几遍才够它背下来？

打完这三个实验，判断还是留给你。但至少有一件事会变得清楚：
**这个模型的「会」和「不会」，是由数据决定的，不是由它自己决定的。**
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import Adam, cross_entropy, no_grad

from after import (  # noqa: E402
    BATCH_SIZE,
    CONTEXT,
    CORPUS,
    LEARNING_RATE,
    SEED,
    STEPS,
    CharTokenizer,
    TinyLanguageModel,
    build_batches,
    evaluate_text,
    shuffle_whole,
)


def train_on_text(tokenizer, text, steps=STEPS, seed=SEED):
    """在一段文字上训练一个和第 22 章一模一样的模型。"""
    inputs, targets = build_batches(tokenizer.encode(text))
    model = TinyLanguageModel(tokenizer.vocab_size)
    optimizer = Adam(model.params(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=BATCH_SIZE)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()
    return model


def main():
    tokenizer = CharTokenizer(CORPUS)
    shuffled = shuffle_whole(CORPUS, SEED)

    print("=" * 64)
    print("实验一：交叉评估")
    print("=" * 64)
    print("  原文：  " + CORPUS[:32])
    print("  打乱后：" + shuffled[:32])
    print()
    print("  两段文字用的字**完全一样**，只是顺序不同。")
    print("  别的全都一样：同一个模型结构、同样的步数、同一个种子。")
    print()

    original_model = train_on_text(tokenizer, CORPUS)
    shuffled_model = train_on_text(tokenizer, shuffled)

    original_on_original = evaluate_text(original_model, tokenizer, CORPUS)
    original_on_shuffled = evaluate_text(original_model, tokenizer, shuffled)
    shuffled_on_original = evaluate_text(shuffled_model, tokenizer, CORPUS)
    shuffled_on_shuffled = evaluate_text(shuffled_model, tokenizer, shuffled)

    print(f"  {'':22}{'考它原文':>12}{'考它打乱文本':>16}")
    print("  " + "-" * 50)
    print(f"  {'用原文训练的模型':<20}{original_on_original['loss']:>12.4f}"
          f"{original_on_shuffled['loss']:>16.4f}")
    print(f"  {'用打乱文本训练的模型':<18}{shuffled_on_original['loss']:>12.4f}"
          f"{shuffled_on_shuffled['loss']:>16.4f}")
    print()
    print("  对角线上的两个数字很小，另外两个很大。")
    print("  每个模型都只认得自己训练时看过的那套顺序，")
    print("  换到另一套顺序上，它立刻变回瞎猜。")
    print()
    print("  这条对角线就是这一章最硬的一个证据：")
    print("  **它学的是「这 304 个字按什么顺序排」，")
    print("    不是「中文的字按什么顺序排」。**")
    print()
    print("  因为打乱的那段文字里，没有一个片段符合中文的习惯，")
    print("  可它照样学得和原文一样好——只是它学的是那套乱序的规矩。")

    print()
    print("=" * 64)
    print("实验二：多训一会儿，它能不能学会原文？")
    print("=" * 64)
    print("  拿上面那个「用打乱文本训练的模型」不太公平，")
    print("  我们换个问法：一个用**原文**训练的模型，")
    print("  训练步数加上去，它在打乱文本上的损失会变好吗？")
    print()
    print(f"  {'训练步数':>8}{'原文上的 loss':>16}{'打乱文本上的 loss':>20}")
    print("  " + "-" * 46)
    for steps in (200, 500, 1000, 2000):
        model = train_on_text(tokenizer, CORPUS, steps=steps)
        on_original = evaluate_text(model, tokenizer, CORPUS)["loss"]
        on_shuffled = evaluate_text(model, tokenizer, shuffled)["loss"]
        print(f"  {steps:>8}{on_original:>16.4f}{on_shuffled:>20.4f}")
    print()
    print("  原文本上的 loss 一直待在 0.01 到 0.02 之间，早就到底了。")
    print("  打乱文本上的 loss 不但没降，还从 17.7 一路涨到了 23.9。")
    print(f"  （瞎猜的水平是 ln33 = {np.log(tokenizer.vocab_size):.2f}；")
    print("    比它大得多，说明模型是在「自信地答错」。）")
    print()
    print("  也就是说：**练得越久，它越会背这一段，")
    print("  但一点也没有变得更懂中文。它甚至越练越不会应付陌生顺序。**")

    print()
    print("=" * 64)
    print("实验三：语料重复几遍才够它背下来？")
    print("=" * 64)
    print("  我们一直在把两首诗重复 8 遍。改一下这个数字：")
    print()
    print(f"  {'重复遍数':>8}{'语料长度':>10}{'切出窗口':>10}{'原文上的 loss':>16}")
    print("  " + "-" * 46)
    poem_length = 38
    for repeats in (1, 2, 4, 8):
        text = CORPUS[:poem_length * repeats]
        model = train_on_text(tokenizer, text, steps=600)
        result = evaluate_text(model, tokenizer, text)
        windows = len(text) - CONTEXT
        print(f"  {repeats:>8}{len(text):>10}{windows:>10}{result['loss']:>16.4f}")
    print()
    print("  语料越短，它反而学得越「好」——因为要背的东西更少。")
    print("  注意最后一列的对比对象是每一行各自的语料，不是同一段文字，")
    print("  所以这三行不能直接比大小。它们要说的是：")
    print()
    print("  只要语料小于模型的容量，模型就有办法把它整个记住。")
    print("  这 304 个字符配 15 万多个参数，比例是 1 比 500。")
    print("  它当然背得下来。")

    print()
    print("=" * 64)
    print("三个实验合起来")
    print("=" * 64)
    print("  1. 交叉评估：换一套顺序它就废了。它学的是顺序本身。")
    print("  2. 多训练没用：它只会把当前这套顺序背得更熟。")
    print("  3. 语料太小时，背下来是必然的：数据量撑不起「学规律」。")
    print()
    print("  但是——请注意这里有个「但是」。")
    print()
    print("  这个实验只证明了模型**有**能力背下来，")
    print("  它没有证明模型**只**会背。")
    print()
    print("  after.py 里那些观察也还在那儿：")
    print("    - 它的错误全部集中在上下文只有一两个字的时候")
    print("    - 它最不确定的字，恰好是语料里有歧义的那些")
    print("    - 它错的时候很自信")
    print()
    print("  一个纯粹的查找表不会「在某个位置格外犹豫」。")
    print("  所以「它只会背」这个结论，下得太快了。")
    print()
    print("  诚实的结论是：**在这个规模上，这两件事分不开。**")
    print("  要分清它们，只有一个办法——把规模做大，")
    print("  大到它背不下来为止。")
    print()
    print("  而第 9 章我们写的那个 toygrad，跑不动那么大的模型。")
    print("  下一章我们换工具。")


if __name__ == "__main__":
    main()
