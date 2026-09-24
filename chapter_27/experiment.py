"""第 27 章 experiment.py —— 我们和 GPT 之间的那三件事，各值多少。

  一、浅层 + 正常学习率：三件事换来换去，loss 几乎一样
  二、深层 + 大学习率：三件事也都换上去，还是几乎一样；
      真正要命的是"归一化放在哪"（那一条我们第 18 章就做对了）
  三、为什么会这样：量一量梯度传到最底下还剩多少

跑法：

    ./.venv/bin/python chapter_27/experiment.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    TRAIN_DATA,
    VAL_DATA,
    VOCAB_SIZE,
    GPTModel,
    evaluate,
    get_batch,
)

BLOCK_SIZE = 48
BATCH_SIZE = 32


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def build_variant(n_layer, learned_pos=False, activation="relu", final_ln=False,
                  pre_ln=True, seed=0):
    """同一个模型，四个开关分别控制那几件事。

    默认值（都关掉）= 第 22 章那个模型。
    """
    torch.manual_seed(seed)
    return GPTModel(
        VOCAB_SIZE, dim=64, n_layer=n_layer, n_head=4, block_size=BLOCK_SIZE,
        pre_ln=pre_ln, learned_pos=learned_pos, final_ln=final_ln,
        activation=F.gelu if activation == "gelu" else F.relu,
    )


def train(model, steps, lr, seed=0, checkpoints=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    trajectory = []
    for step in range(steps):
        x, y = get_batch(TRAIN_DATA, BLOCK_SIZE, BATCH_SIZE, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % (steps // checkpoints) == 0:
            trajectory.append(loss.item())
    return trajectory


# ---------------------------------------------------------------- 实验一

def experiment_one():
    print("=" * 74)
    print("一、4 层 + 学习率 0.003：三件事各换一个，训练 150 步")
    print("=" * 74)
    print("每一行只改一个地方，其它两个保持我们第 22 章的做法。")
    print()

    variants = [
        ("我们的全部选择（ReLU + 固定位置 + 输出层前不加 LN）", dict()),
        ("只换成 GELU", dict(activation="gelu")),
        ("只换成可学习的位置表", dict(learned_pos=True)),
        ("只加上输出层前那一次 LN", dict(final_ln=True)),
        ("GPT 的全部选择", dict(activation="gelu", learned_pos=True, final_ln=True)),
    ]

    print(pad("配置", 44) + pad("训练 loss", 12) + pad("验证 loss", 12) + "参数量")
    print("-" * 74)
    for name, options in variants:
        model = build_variant(4, **options)
        trajectory = train(model, steps=150, lr=3e-3)
        val = evaluate(model, VAL_DATA, BLOCK_SIZE, BATCH_SIZE)
        print(pad(name, 44) + pad(f"{trajectory[-1]:.3f}", 12)
              + pad(f"{val:.3f}", 12) + f"{model.n_params():,}")
    print()
    print("五个数字几乎一模一样。这三件事**不是**什么新模块 ——")
    print("在浅层、学习率正常的时候，换不换都看不出来。")
    print()


# ---------------------------------------------------------------- 实验二

def experiment_two():
    print("=" * 74)
    print("二、12 层 + 学习率 0.01：这里能看出区别的只有一件事")
    print("=" * 74)
    print(pad("配置", 40) + pad("第 1/4", 10) + pad("第 2/4", 10)
          + pad("第 3/4", 10) + pad("第 4/4", 10) + "验证 loss")
    print("-" * 74)
    variants = [
        ("我们的全部选择（= 第 22 章的模型）", dict()),
        ("我们的 + GELU + 可学习位置 + 最后那次 LN", dict(activation="gelu",
                                                 learned_pos=True, final_ln=True)),
        ("归一化挪到残差外面（post-LN）", dict(pre_ln=False)),
    ]
    for name, options in variants:
        model = build_variant(12, **options)
        start = time.time()
        trajectory = train(model, steps=150, lr=1e-2)
        val = evaluate(model, VAL_DATA, BLOCK_SIZE, BATCH_SIZE)
        print(pad(name, 40) + "".join(pad(f"{loss:.3f}", 10) for loss in trajectory)
              + f"{val:.3f}   （{time.time() - start:.0f} 秒）")
    print()
    print("前两行挨在一起（0.47 上下），第三行从头到尾停在 3.17 ——")
    print("也就是说：这三件事换不换，我们和 GPT 打平；")
    print("而第 18 章那个「先归一化再算」，才是 12 层能不能训起来的分水岭。")
    print()


# ---------------------------------------------------------------- 实验三

def experiment_three():
    print("=" * 74)
    print("三、为什么会这样：梯度传到最底下还剩多少")
    print("=" * 74)
    print("词嵌入是最底下的一层，它的梯度要穿过 12 个 block 才能到手上，")
    print("所以它的梯度范数（grad.norm()）就是'信号还剩多少'的温度计。")
    print()
    print(pad("配置", 34) + pad("第 1 步", 12) + pad("第 10 步", 12)
          + pad("第 20 步", 12) + "第 30 步")
    print("-" * 74)

    trained = {}
    for name, pre_ln in (("我们的写法（先归一化再算）", True),
                         ("post-LN（先算再归一化）", False)):
        model = build_variant(12, pre_ln=pre_ln)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        rng = np.random.default_rng(0)
        marks = {1: None, 10: None, 20: None, 30: None}
        for step in range(30):
            x, y = get_batch(TRAIN_DATA, BLOCK_SIZE, BATCH_SIZE, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if step + 1 in marks:
                marks[step + 1] = model.tok_emb.weight.grad.norm().item()
            optimizer.step()
        print(pad(name, 34) + "".join(pad(f"{marks[s]:.1e}", 12) for s in (1, 10, 20, 30)))
        trained[name] = model
    print()
    print("再看一眼每个 block 拿到的梯度（12 层，训练 30 步之后）：")
    print()
    print(pad("配置", 34) + pad("第 1 层", 12) + pad("第 4 层", 12)
          + pad("第 8 层", 12) + "第 12 层")
    print("-" * 74)
    for name, model in trained.items():
        norms = []
        for index in (0, 3, 7, 11):
            block = model.blocks[index]
            total = sum((p.grad ** 2).sum() for p in block.parameters() if p.grad is not None)
            norms.append(torch.sqrt(total).item())
        print(pad(name, 34) + "".join(pad(f"{value:.1e}", 12) for value in norms))
    print()
    print("（第 1 层是最底下的那一层，第 12 层紧挨着输出。）")
    print()
    print("post-LN 那一行：底层梯度从 5.0e-02 掉到 4.5e-13（十几个数量级），")
    print("最底下的层收不到任何信号，模型只剩下最上面几层在动。")
    print("这不是'更聪明'和'不够聪明'的区别，是'能训'和'不能训'的区别 ——")
    print("而这一条，我们第 18 章就选对了。")
    print()


if __name__ == "__main__":
    experiment_one()
    experiment_two()
    experiment_three()
