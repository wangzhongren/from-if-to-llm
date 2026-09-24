"""第 24 章 before：用眼睛评估，然后发现眼睛不够用。

前面几章我们判断模型好不好的办法一直是同一个：

    让它写一段，我们自己读一读，觉得「像中文」就算好。

这个办法真的能用吗？这个文件就认真做一次。

结果是这样的：

  给它一个语料里的开头，它写出来的东西漂亮得让人吃惊。
  我们会想："它学会了。"

  给它一个语料外的开头，它写出来的东西很快就散了。
  我们会想："它什么都没学会。"

  两个结论互相矛盾。也就是说——**只看它写出来的东西，判断不了。**

不是因为它写得不好。是因为我们只看了两个样本，就急着下结论。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from after import (  # noqa: E402
    CORPUS,
    CharTokenizer,
    generate_greedy,
    train_default_model,
)


def show(model, tokenizer, prompt, length=30):
    """人眼评估：让它写，我们看。"""
    text = generate_greedy(model, tokenizer, prompt, length)
    print(f"  开头：{prompt}")
    print(f"  它写：{text}")
    print()
    return text


def main():
    tokenizer = CharTokenizer(CORPUS)
    print("=" * 60)
    print("准备")
    print("=" * 60)
    model, _, _ = train_default_model(tokenizer)
    print("  模型是第 22 章那个，一个字都没改。")

    print()
    print("=" * 60)
    print("第一印象：它学会了")
    print("=" * 60)
    print("  我们给它几个语料里的开头，看它写什么：")
    print()
    show(model, tokenizer, "床前")
    show(model, tokenizer, "鹅鹅鹅")
    show(model, tokenizer, "白毛浮绿水")
    print("  三段都通顺。「明月光」「曲项向天歌」「红掌拨清波」——")
    print("  字对，顺序对，连诗的节奏都对。")
    print("  看到这个，第一反应一定是：它学会了中文。")

    print()
    print("=" * 60)
    print("换个开头：它什么都没学会")
    print("=" * 60)
    print("  现在换几个语料里没有的开头。用的还是那 33 个字：")
    print()
    for prompt in ["故乡故乡故乡故乡故乡故乡故乡故乡",
                   "水水水水水水水水水水水水水水水水",
                   "月月月月月月月月月月月月月月月月"]:
        show(model, tokenizer, prompt, 26)
    print("  前面还像话，越写越乱，最后完全散了。")
    print("  看到这个，第一反应也一定是：它根本什么都没学会，")
    print("  只是在复读训练数据。")

    print()
    print("=" * 60)
    print("两个结论，互相矛盾")
    print("=" * 60)
    print("  同一批权重、同一个模型。")
    print("  换个开头，我们的判断就从「学会了」变成「什么都没学会」。")
    print()
    print("  问题出在哪？")
    print("  出在「我们自己读一读」这个办法本身。")
    print()
    print("    - 它一次只给我们一个样本。")
    print("    - 我们没有量过它到底有多确定。")
    print("    - 我们也没有量过它有多少答案是背出来的。")
    print("    - 我们手里没有一把尺子，只有一双眼睛。")
    print()
    print("  眼睛能看出来的东西，只到「像不像中文」为止。")
    print("  而这一章要问的问题比这个细：")
    print("    - 它到底是学会了「中文怎么排」，还是背下了「这 304 个字怎么排」？")
    print("    - 它什么时候最没把握？")
    print("    - 它猜错的时候，自己知道吗？")
    print()
    print("  这三个问题，眼睛一个都回答不了。")
    print()
    print("  下一节我们换一把尺子。")
    print("  这把尺子不读它写出来的东西，")
    print("  而是直接问它：**下一个字，你觉得是哪一个？**")
    print("  然后跟正确答案比。")


if __name__ == "__main__":
    main()
