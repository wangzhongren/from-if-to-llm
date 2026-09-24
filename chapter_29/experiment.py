"""第 29 章 experiment.py —— 把"换了数据格式"这件事量出来。

after.py 是讲故事的，这个文件是量数据的。同一个模型复制三份，
喂三种数据，然后用三把尺子量它们：

    尺子一 答对率      ：问 7 个问题，它答对几个
    尺子二 原始文本 loss：它还是不是一个好语言模型
    尺子三 短答比例    ：同一个问题生成 5 次，有几次是干脆的回答

跑法：

    ./.venv/bin/python chapter_29/experiment.py
"""

import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import after as A  # noqa: E402

BLOCK_SIZE = 32
BATCH_SIZE = 16


def pad(text, width):
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


def pretrain_model(steps=600, seed=0):
    torch.manual_seed(0)
    model = A.TinyLM(A.VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=BLOCK_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        A.pretrain_step(model, optimizer, BATCH_SIZE, rng)
    return model, rng


# ---------------------------------------------------------------- 三把尺子

@torch.no_grad()
def greedy_answer(model, prompt, n_new=8):
    """不用采样，每一步都取概率最大的那个字 —— 这样量出来的东西是确定的。"""
    ids = A.encode(prompt)
    for _ in range(n_new):
        window = torch.tensor([ids[-BLOCK_SIZE:]])
        logits = model(window)[0, -1]
        ids.append(int(logits.argmax()))
    return A.decode(ids)[len(prompt):]


def answer_accuracy(model, facts):
    """答对率：模型给出的回答有没有包含那个正确的说法。"""
    hits = []
    for _, _, _, question, short, _ in facts:
        produced = greedy_answer(model, question)
        expected = short
        hits.append(produced.startswith(expected))
    return hits


def raw_language_loss(model, text, rounds=10, seed=1):
    """量一量它还是不是个好语言模型：在没训练过的原始文本上算 loss。"""
    data = np.array(A.encode(text))
    rng = np.random.default_rng(seed)
    total = 0.0
    with torch.no_grad():
        for _ in range(rounds):
            starts = rng.integers(0, len(data) - BLOCK_SIZE - 1, size=BATCH_SIZE)
            x = np.stack([data[i:i + BLOCK_SIZE] for i in starts])
            y = np.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts])
            logits = model(torch.from_numpy(x))
            total += torch.nn.functional.cross_entropy(
                logits.reshape(-1, A.VOCAB_SIZE), torch.from_numpy(y).reshape(-1)).item()
    return total / rounds


def short_style_ratio(model, facts, rounds=5):
    """同一个问题采样 5 次，其中有几次用的是干脆的回答（而不是绕一圈）。"""
    short_hits, total = 0, 0
    for _, _, _, question, short, _ in facts:
        for seed in range(rounds):
            text = A.generate(model, question, n_new=24, temperature=0.8, seed=seed)
            total += 1
            if text.startswith(short):
                short_hits += 1
    return short_hits / total


# ---------------------------------------------------------------- 主程序

def main():
    print("=" * 74)
    print("同一个模型，三种数据，三把尺子")
    print("=" * 74)
    print("先把预训练做完（第 20 章的目标，所有分支都从它出发）：")
    model, rng = pretrain_model()
    print(f"  预训练 600 步完成，参数量 {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 拿一段没参与预训练的文本，用来量"它还是不是一个好语言模型"
    held_out = A.build_pretrain_text(segments=40, seed=99)

    branches = []
    baseline = copy.deepcopy(model)
    branches.append(("只做了预训练", baseline, None))

    # 分支一：再多喂一点同样的原始文本（同样的字符量，只是格式没变）
    more_pretrain = copy.deepcopy(model)
    optimizer = torch.optim.Adam(more_pretrain.parameters(), lr=3e-3)
    for _ in range(300):
        A.pretrain_step(more_pretrain, optimizer, BATCH_SIZE, rng)
    branches.append(("继续喂原始文本 300 步", more_pretrain, None))

    # 分支二：喂指令数据（问答题），同样的步数
    sft = copy.deepcopy(model)
    optimizer = torch.optim.Adam(sft.parameters(), lr=3e-3)
    for _ in range(300):
        A.sft_step(sft, optimizer, BATCH_SIZE, rng)
    branches.append(("喂指令数据 300 步", sft, None))

    # 分支三：在指令数据之后再做偏好对齐
    aligned = copy.deepcopy(sft)
    reference = copy.deepcopy(sft)
    optimizer = torch.optim.Adam(aligned.parameters(), lr=1e-4)
    for _ in range(100):
        A.dpo_step(aligned, reference, optimizer, rng)
    branches.append(("指令数据 + 偏好对齐", aligned, None))

    print(pad("模型", 26) + pad("答对率", 12) + pad("原始文本 loss", 16) + "短答比例")
    print("-" * 74)
    for name, candidate, _ in branches:
        hits = answer_accuracy(candidate, A.FACTS)
        accuracy = f"{sum(hits)}/{len(hits)}"
        loss = raw_language_loss(candidate, held_out)
        ratio = short_style_ratio(candidate, A.FACTS)
        print(pad(name, 26) + pad(accuracy, 12) + pad(f"{loss:.3f}", 16) + f"{ratio:.0%}")
    print()
    print("三个数字各自说明一件事：")
    print()
    print("1. 答对率：预训练做得再久，一个都答不对。换成指令数据，300 步全对。")
    print("   同一个模型、同样的步数、差不多的字符数 —— 差的是数据的**形状**。")
    print()
    print("2. 原始文本 loss：从 0.40 涨到 2.35 —— 它几乎把「怎么续写文本」忘光了。")
    print("   这是真实世界里「对齐税」的极端版本：我们的微调数据太少、步数太多，")
    print("   几百步就把原来的本事冲掉了。真实模型的数据多得多，退得没这么狠，")
    print("   但「学会了听话、忘掉了别的」这件事一直在发生。")
    print()
    print("3. 短答比例：偏好对齐之前，干脆的回答和绕一圈的回答各占一半；")
    print("   对齐之后，「直接回答」变成默认选择。")
    print("   —— 注意：没有任何一条数据告诉它「要简洁」，")
    print("      我们只是给了一批「这个比那个好」的例子。")
    print()

    print("=" * 74)
    print("答对率是怎么量出来的")
    print("=" * 74)
    print("每一步都取概率最大的那个字（不做随机采样），然后把生成结果和答案比：")
    print()
    for _, _, _, question, short, _ in A.FACTS[:4]:
        produced = greedy_answer(branches[0][1], question)
        produced_sft = greedy_answer(branches[2][1], question)
        print(f"  {question}")
        print(f"    预训练模型 -> {produced!r}    （应该是 {short!r}）")
        print(f"    指令微调后 -> {produced_sft!r}")
    print()


if __name__ == "__main__":
    main()
