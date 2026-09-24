"""第 25 章 experiment.py —— 证明 PyTorch 只是把我们写过的东西封装起来了。

四个实验：

  实验一：同一份权重，两个引擎算出来的分数一模一样。
  实验二：同一个起点、同一份数据、同一个种子，两边各训练 100 步，loss 走势重合。
  实验三：nn.MultiheadAttention（PyTorch 封装好的注意力）和我们手写的多头注意力
          在权重相同时输出一致。
  实验四：同一个模型，两个引擎的速度和内存。

内存必须在独立的进程里量：一个进程的峰值内存只会记住最大值，
两个引擎跑在同一个进程里，第二个引擎永远量不出更小的数。

跑法：

    ./.venv/bin/python chapter_25/experiment.py
"""

import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from after import (  # noqa: E402
    DATA,
    VOCAB_SIZE,
    TorchModel,
    get_batch,
    load_our_weights,
)
from before import OurModel  # noqa: E402
from toygrad import Adam, Tensor, cross_entropy, layer_norm  # noqa: E402

OUR_SIZE = dict(dim=32, n_layer=2, n_head=4, block_size=16)

# 实验四要量的两个模型：(结构参数, 跑多少步)
# 大模型只跑 10 步 —— 我们的引擎每一步都要吃掉上百 MB，跑多了这台机器会受不了
BENCH_SIZES = {
    "80 万参数": (dict(dim=128, n_layer=4, n_head=4, block_size=32), 20),
    "200 万参数": (dict(dim=192, n_layer=6, n_head=4, block_size=32), 10),
}


def make_ours(seed=0, **size):
    return OurModel(VOCAB_SIZE, seed=seed, **size)


def make_torch(seed=0, **size):
    torch.manual_seed(seed)
    return TorchModel(VOCAB_SIZE, **size)


def pad(text, width):
    """中文字符在终端里占两个格子，直接 f-string 对齐会歪。"""
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


# ---------------------------------------------------------------- 实验一

def experiment_one():
    print("=" * 64)
    print("实验一：同一份权重，两个引擎算出同样的分数吗？")
    print("=" * 64)
    ours = make_ours(seed=0, **OUR_SIZE)
    theirs = load_our_weights(make_torch(seed=0, **OUR_SIZE), ours)

    rng = np.random.default_rng(42)
    x, y = get_batch(DATA, block_size=16, batch_size=8, rng=rng)

    with torch.no_grad():
        logits_torch = theirs(x).numpy()
    logits_ours = ours.forward(x.numpy()).data

    print(f"输入形状 {tuple(x.shape)}，输出形状 {logits_torch.shape}（每个位置 36 个分数）")
    print(f"两条输出的最大差值：{np.abs(logits_torch - logits_ours).max():.2e}")
    print("（差的这一点来自 float32 和 float64 的舍入，不是算错了）")
    print()


# ---------------------------------------------------------------- 实验二

def experiment_two():
    print("=" * 64)
    print("实验二：同一起点、同一份数据、同一个种子，各训练 100 步")
    print("=" * 64)
    ours = make_ours(seed=0, **OUR_SIZE)
    theirs = load_our_weights(make_torch(seed=0, **OUR_SIZE), ours)

    ours_opt = Adam(ours.parameters(), lr=3e-3)
    theirs_opt = torch.optim.Adam(theirs.parameters(), lr=3e-3)
    ours_rng = np.random.default_rng(0)
    torch_rng = np.random.default_rng(0)      # 两边喂的 batch 必须完全一样

    print(pad("步数", 8) + pad("我们的引擎", 14) + pad("PyTorch", 14) + "差值")
    print("-" * 64)
    for step in range(100):
        x, y = get_batch(DATA, block_size=16, batch_size=8, rng=ours_rng)
        loss_ours = cross_entropy(ours.forward(x.numpy()), y.numpy())
        ours_opt.zero_grad()
        loss_ours.backward()
        ours_opt.step()

        logits = theirs(x)
        loss_theirs = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        theirs_opt.zero_grad()
        loss_theirs.backward()
        theirs_opt.step()

        if (step + 1) % 10 == 0:
            a, b = loss_ours.item(), loss_theirs.item()
            print(pad(str(step + 1), 8) + pad(f"{a:.4f}", 14)
                  + pad(f"{b:.4f}", 14) + f"{abs(a - b):.5f}")
    print()


# ---------------------------------------------------------------- 实验三

def experiment_three():
    print("=" * 64)
    print("实验三：PyTorch 连注意力都封装好了（nn.MultiheadAttention）")
    print("=" * 64)
    our_block = make_ours(seed=0, **OUR_SIZE).params
    # 我们自己写的：四个独立的投影矩阵 Wq/Wk/Wv/Wo（没有偏置，第 14 章就是 x @ W）
    # PyTorch 封装的：把 Q、K、V 三个投影拼成一个大矩阵（in_proj_weight）
    packed = nn.MultiheadAttention(OUR_SIZE["dim"], OUR_SIZE["n_head"],
                                   batch_first=True, bias=False)

    def to_tensor(w):
        return torch.tensor(np.asarray(w), dtype=torch.float32)

    with torch.no_grad():
        packed.in_proj_weight.copy_(torch.cat([
            to_tensor(our_block["block0.Wq"].data.T),
            to_tensor(our_block["block0.Wk"].data.T),
            to_tensor(our_block["block0.Wv"].data.T),
        ], dim=0))
        packed.out_proj.weight.copy_(to_tensor(our_block["block0.Wo"].data.T))

    rng = np.random.default_rng(7)
    x_np, _ = get_batch(DATA, block_size=16, batch_size=4, rng=rng)
    length = x_np.shape[1]
    mask = np.triu(np.ones((length, length), dtype=bool), k=1)

    # 喂给注意力的输入：查表 + 位置编码 + 一次 LayerNorm（第 19 章的写法）
    ours_model = make_ours(seed=0, **OUR_SIZE)
    x_emb_ours = Tensor(our_block["tok_emb"].data[x_np]) + Tensor(ours_model.pos_table[:length])
    x_ln_ours = layer_norm(x_emb_ours, our_block["block0.ln1_w"], our_block["block0.ln1_b"])
    ours_out = ours_model.attention(x_ln_ours, 0, mask).data

    x_emb = to_tensor(our_block["tok_emb"].data[x_np]) + to_tensor(ours_model.pos_table[:length])
    x_ln = torch.nn.functional.layer_norm(
        x_emb, (OUR_SIZE["dim"],),
        weight=to_tensor(our_block["block0.ln1_w"].data),
        bias=to_tensor(our_block["block0.ln1_b"].data),
    )
    with torch.no_grad():
        theirs_out = packed(x_ln, x_ln, x_ln, attn_mask=torch.tensor(mask),
                            need_weights=False)[0].numpy()

    print(f"两条输出的最大差值：{np.abs(theirs_out - ours_out).max():.2e}")
    print("我们写的几十行，变成了一个 nn.MultiheadAttention(...)。里面做的事没有变：")
    print("三个投影 -> 拆头 -> 打分 -> 掩码 -> softmax -> 加权求和 -> 拼回来 -> 输出投影。")
    print()


# ---------------------------------------------------------------- 实验四

def bench_child():
    """在独立进程里量一个引擎的速度和内存。

    内存要看"训练前后涨了多少"，而不是进程峰值：光是 import torch 就要占掉几百 MB，
    那个数字和模型无关。
    """
    import resource

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    engine, size_name = sys.argv[2], sys.argv[3]
    size, steps = BENCH_SIZES[size_name]
    rng = np.random.default_rng(0)
    baseline = rss()
    start = time.time()
    if engine == "toygrad":
        model = make_ours(seed=0, **size)
        opt = Adam(model.parameters(), lr=3e-3)
        for _ in range(steps):
            x, y = get_batch(DATA, block_size=32, batch_size=16, rng=rng)
            loss = cross_entropy(model.forward(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        n_params = model.n_params()
    else:
        model = make_torch(seed=0, **size)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(steps):
            x, y = get_batch(DATA, block_size=32, batch_size=16, rng=rng)
            loss = torch.nn.functional.cross_entropy(
                model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        n_params = model.n_params()
    seconds = (time.time() - start) / steps
    print(f"{engine}|{n_params}|{seconds * 1000:.0f}|{rss() - baseline:.0f}|{steps}")


def experiment_four():
    print("=" * 64)
    print("实验四：同一个模型，两个引擎跑同样的步数，比时间和内存")
    print("=" * 64)
    print(pad("模型", 20) + pad("参数量", 14) + pad("引擎", 14)
          + pad("毫秒/步", 12) + "训练涨的内存")
    print("-" * 64)
    for size_name in BENCH_SIZES:
        results = {}
        for engine in ("toygrad", "torch"):
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--bench", engine, size_name],
                capture_output=True, text=True, check=True)
            _, n_params, ms, memory, steps = out.stdout.strip().split("|")
            results[engine] = (int(n_params), float(ms), float(memory))

        size = BENCH_SIZES[size_name][0]
        label = f"dim={size['dim']}，{size['n_layer']} 层"
        for engine, name in (("toygrad", "我们的引擎"), ("torch", "PyTorch")):
            n_params, ms, memory = results[engine]
            print(pad(label, 20) + pad(f"{n_params:,}", 14) + pad(name, 14)
                  + pad(f"{ms:.0f}", 12) + f"{memory:.0f} MB")
        ms_ours, ms_theirs = results["toygrad"][1], results["torch"][1]
        mem_ours, mem_theirs = results["toygrad"][2], results["torch"][2]
        print(pad("", 20) + pad("", 14) + pad("倍数", 14) + pad(f"{ms_ours / ms_theirs:.1f}x", 12)
              + f"省下 {(1 - mem_theirs / mem_ours) * 100:.0f}%")
        print("-" * 64)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--bench":
        bench_child()
    else:
        experiment_one()
        experiment_two()
        experiment_three()
        experiment_four()
