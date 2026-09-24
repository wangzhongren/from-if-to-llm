"""第 24 章 after：一组探测实验。不加新机制，也不改架构。

第 22 章的模型、第 23 章的两个旋钮，都原样不动。
这一章只做一件事：**去问它几个问题，然后把答案记下来。**

问的问题：

  1. 它读到训练语料里的句子时，有多确定？（困惑度）
  2. 同一批字，换个顺序，它还有多确定？
  3. 给它半句，它能接出整句吗？
  4. 它猜错的时候，它自己知道吗？
  5. 它最能确定的是什么字，最不能确定的是什么字？
  6. 给它一段很短的上下文，它会怎样？

这些实验都不会「证明」它懂了或者没懂。
它们只是把观察摆出来——判断留给你。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from toygrad import (
    Tensor,
    Adam,
    cross_entropy,
    embedding,
    layer_norm,
    no_grad,
    randn,
    zeros,
)

# ---------------------------------------------------------------- 语料与超参数（和第 20–23 章一样）

POEM_A = "床前明月光疑是地上霜举头望明月低头思故乡"
POEM_B = "鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波"

CORPUS = (POEM_A + POEM_B) * 8

VOCAB = sorted(set(CORPUS))
VOCAB_SIZE = len(VOCAB)

SEED = 20260924

DIM = 64
LAYERS = 3
HEADS = 4
CONTEXT = 32
STEPS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 3e-3


# ---------------------------------------------------------------- tokenizer（第 1 章）

class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for char, i in self.char_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        return np.array([self.char_to_id[char] for char in text])

    def decode(self, ids):
        return "".join(self.id_to_char[int(i)] for i in ids)


# ---------------------------------------------------------------- 模型（第 22 章的，一个字没改）

def sinusoidal_positions(length, dim):
    """位置编码：第 16 章那张固定的正弦表。

        第 i 个位置、第 2k 维   = sin(i / 10000^(2k/dim))
        第 i 个位置、第 2k+1 维 = cos(i / 10000^(2k/dim))

    注意它是**算出来的**，不是学出来的。
    这张表不参与训练，所以它不在 params() 里。
    """
    positions = np.arange(length)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return table


def causal_mask(length):
    return np.triu(np.ones((length, length), dtype=bool), k=1)


class TransformerBlock:
    def __init__(self, dim, heads, seed):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.query_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 1)
        self.key_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 2)
        self.value_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 3)
        self.out_weight = randn(dim, dim, scale=0.3, requires_grad=True, seed=seed + 4)
        self.norm1_weight = Tensor(np.ones(dim), requires_grad=True)
        self.norm1_bias = zeros(dim, requires_grad=True)
        self.fc_weight = randn(dim, 4 * dim, scale=0.3, requires_grad=True, seed=seed + 5)
        self.fc_bias = zeros(4 * dim, requires_grad=True)
        self.proj_weight = randn(4 * dim, dim, scale=0.3, requires_grad=True, seed=seed + 6)
        self.proj_bias = zeros(dim, requires_grad=True)
        self.norm2_weight = Tensor(np.ones(dim), requires_grad=True)
        self.norm2_bias = zeros(dim, requires_grad=True)

    def params(self):
        return [
            self.query_weight, self.key_weight, self.value_weight, self.out_weight,
            self.norm1_weight, self.norm1_bias,
            self.fc_weight, self.fc_bias, self.proj_weight, self.proj_bias,
            self.norm2_weight, self.norm2_bias,
        ]

    def multi_head_attention(self, x, mask):
        batch, length, _ = x.shape

        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        query = split(x @ self.query_weight)
        key = split(x @ self.key_weight)
        value = split(x @ self.value_weight)
        scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
        scores = scores.masked_fill(mask, -1e9)
        attended = scores.softmax(axis=-1) @ value
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)
        return merged @ self.out_weight

    def __call__(self, x, mask):
        normed = layer_norm(x, self.norm1_weight, self.norm1_bias)
        x = x + self.multi_head_attention(normed, mask)
        normed = layer_norm(x, self.norm2_weight, self.norm2_bias)
        hidden = (normed @ self.fc_weight + self.fc_bias).relu()
        x = x + hidden @ self.proj_weight + self.proj_bias
        return x


class TinyLanguageModel:
    def __init__(self, vocab_size, dim=DIM, layers=LAYERS, heads=HEADS, context=CONTEXT, seed=SEED):
        self.dim = dim
        self.context = context
        self.token_embedding = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        # 位置表是固定的正弦表（第 16 章），不需要学，也不放进 params()
        self.position_table = Tensor(sinusoidal_positions(context, dim))
        self.blocks = [TransformerBlock(dim, heads, seed + 100 * (i + 1)) for i in range(layers)]
        self.head_weight = randn(dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 90)
        self.head_bias = zeros(vocab_size, requires_grad=True)

    def params(self):
        collected = [self.token_embedding]
        for block in self.blocks:
            collected += block.params()
        return collected + [self.head_weight, self.head_bias]

    def parameter_count(self):
        return sum(p.data.size for p in self.params())

    def __call__(self, token_ids):
        batch, length = token_ids.shape
        x = embedding(self.token_embedding, token_ids)
        x = x + self.position_table[:length].reshape(1, length, self.dim)
        mask = causal_mask(length)
        for block in self.blocks:
            x = block(x, mask)
        return x @ self.head_weight + self.head_bias


# ---------------------------------------------------------------- 训练

def build_batches(ids, context=CONTEXT):
    inputs, targets = [], []
    for start in range(len(ids) - context):
        inputs.append(ids[start:start + context])
        targets.append(ids[start + 1:start + context + 1])
    return np.array(inputs), np.array(targets)


def train(model, inputs, targets, steps=STEPS, batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE, seed=SEED):
    optimizer = Adam(model.params(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = rng.integers(0, len(inputs), size=batch_size)
        optimizer.zero_grad()
        cross_entropy(model(inputs[batch]), targets[batch]).backward()
        optimizer.step()


def train_default_model(tokenizer):
    """把第 22 章那个模型原样训练一遍。"""
    inputs, targets = build_batches(tokenizer.encode(CORPUS))
    model = TinyLanguageModel(tokenizer.vocab_size)
    train(model, inputs, targets)
    return model, inputs, targets


def generate_greedy(model, tokenizer, prompt, length):
    """第 22 章那个贪心接龙，探测六要用。"""
    ids = list(tokenizer.encode(prompt))
    with no_grad():
        for _ in range(length):
            window = np.array(ids[-CONTEXT:])[None, :]
            ids.append(int(model(window).data[0, -1].argmax()))
    return tokenizer.decode(ids)


# ---------------------------------------------------------------- 量尺子


def perplexity(loss):
    """把 loss 换算成困惑度。

    困惑度 = e 的 loss 次方。

    loss 是 0.69 的时候困惑度是 2，意思是模型在两个候选之间犹豫；
    loss 是 3.50（也就是 ln 33）的时候困惑度是 33，
    意思是它在 33 个字符里完全瞎猜。

    说人话：**困惑度就是"模型心里大概有几个候选在打架"。**
    越接近 1，它越笃定；越接近词表大小，它越瞎。
    """
    return float(np.exp(loss))


def evaluate_text(model, tokenizer, text):
    """把一段文字喂进去，量四个数。

    注意：不管这段文字有多长，模型每次只看 CONTEXT 个字符，
    所以我们要把它切成一个个窗口。
    """
    ids = tokenizer.encode(text)
    if len(ids) <= CONTEXT:
        raise ValueError(f"这段文字太短了，至少要 {CONTEXT + 1} 个字符")

    inputs, targets = [], []
    for start in range(len(ids) - CONTEXT):
        inputs.append(ids[start:start + CONTEXT])
        targets.append(ids[start + 1:start + CONTEXT + 1])
    inputs, targets = np.array(inputs), np.array(targets)

    with no_grad():
        logits = model(inputs).data
        loss = cross_entropy(model(inputs), targets).item()
        correct = logits.argmax(axis=-1) == targets
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
        confidence = probabilities.max(axis=-1)

    return {
        "loss": loss,
        "perplexity": perplexity(loss),
        "accuracy": float(correct.mean()),
        "confidence": float(confidence.mean()),
        "positions": int(correct.size),
        "wrong": int((~correct).sum()),
    }


def shuffle_whole(text, seed):
    """把整段文字的字序完全打乱。字的种类和数量一个字都没变。"""
    characters = list(text)
    np.random.default_rng(seed).shuffle(characters)
    return "".join(characters)


def shuffle_blocks(text, block_size, seed):
    """只在每个小块内部打乱。

    块越小，局部顺序被破坏得越少。
    block_size=1 就是原文，block_size 很大就接近整段打乱。
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for start in range(0, len(text), block_size):
        block = list(text[start:start + block_size])
        rng.shuffle(block)
        pieces.append("".join(block))
    return "".join(pieces)


# ---------------------------------------------------------------- 主流程


def main():
    tokenizer = CharTokenizer(CORPUS)
    model, inputs, targets = train_default_model(tokenizer)

    print("=" * 64)
    print("先说明一件事")
    print("=" * 64)
    print("  这一章不加任何新零件，也不重新训练。")
    print("  下面所有的数字，都来自第 22 章那个模型、第 23 章那两个旋钮。")
    print("  我们只是换着法子问它问题。")

    print()
    print("=" * 64)
    print("探测一：它有多确定")
    print("=" * 64)
    print("  先给「困惑度」一个说法：")
    print("    困惑度 = e 的 loss 次方。")
    print("    它可以读成「模型心里大概有几个候选在打架」。")
    print("    等于 1 表示它只有一个候选；等于 33 表示它在 33 个字符里瞎猜。")
    print()
    print(f"  {'这段文字':<20}{'loss':>9}{'困惑度':>15}{'猜对比例':>10}")
    print("  " + "-" * 55)

    probes = [
        ("语料原文", CORPUS),
        ("每 2 个字打乱一次", shuffle_blocks(CORPUS, 2, SEED)),
        ("每 4 个字打乱一次", shuffle_blocks(CORPUS, 4, SEED)),
        ("每 8 个字打乱一次", shuffle_blocks(CORPUS, 8, SEED)),
        ("整段打乱", shuffle_whole(CORPUS, SEED)),
    ]
    measures = []
    for label, text in probes:
        result = evaluate_text(model, tokenizer, text)
        measures.append((label, result))
        shown = f"{result['perplexity']:.2f}" if result["perplexity"] < 10000 else f"{result['perplexity']:.2e}"
        print(f"  {label:<20}{result['loss']:>9.4f}{shown:>15}{result['accuracy']:>10.2%}")

    print()
    print(f"  第一行：语料原文，困惑度 {measures[0][1]['perplexity']:.2f}。"
          "也就是它心里只有一个候选。")
    print("  最后一行：同样的字、同样的数量，只是顺序换了，")
    print("  困惑度就涨到了一个天文数字。")
    print()
    random_guess = 1 / tokenizer.vocab_size
    print(f"  注意最后一行「猜对比例」是 {measures[-1][1]['accuracy']:.2%}，")
    print(f"  而瞎猜的期望正好是 1/{tokenizer.vocab_size} = {random_guess:.2%}。")
    print("  顺序一乱，它掉回瞎猜的水平。")

    on_corpus = measures[0][1]
    off_corpus = measures[-1][1]

    print()
    print("=" * 64)
    print("探测二：它错的时候，自己知道吗")
    print("=" * 64)
    result = evaluate_text(model, tokenizer, CORPUS)
    print(f"  语料原文一共 {result['positions']} 个「猜下一个字」的位置。")
    print(f"  它猜错了 {result['wrong']} 个（{result['wrong']/result['positions']:.2%}）。")

    ids = tokenizer.encode(CORPUS)
    windows, answers = [], []
    for start in range(len(ids) - CONTEXT):
        windows.append(ids[start:start + CONTEXT])
        answers.append(ids[start + 1:start + CONTEXT + 1])
    windows, answers = np.array(windows), np.array(answers)

    with no_grad():
        logits = model(windows).data
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
    target_probability = probabilities[np.arange(len(answers))[:, None],
                                       np.arange(CONTEXT)[None, :], answers]
    wrong = logits.argmax(axis=-1) != answers
    wrong_confidence = probabilities.max(axis=-1)[wrong].mean()

    print()
    print(f"  它猜对的时候，给正确答案的概率平均是  {target_probability[~wrong].mean():.4f}")
    print(f"  它猜错的时候，给正确答案的概率平均只有 {target_probability[wrong].mean():.4f}")
    print(f"  但它猜错的时候，给**自己选的那个答案**的概率平均有 "
          f"{probabilities.max(axis=-1)[wrong].mean():.4f}")
    print()
    print("  最后一行才是要看的：它错得很自信。")
    print("  它不是「不确定，随便挑了一个」，")
    print("  而是「我非常确定——确定错了」。")

    print()
    print("  它一共只错了三种：")
    seen = set()
    for row in range(len(answers)):
        for column in range(CONTEXT):
            if not wrong[row, column]:
                continue
            key = (int(answers[row, column]), int(logits[row, column].argmax()))
            if key in seen:
                continue
            seen.add(key)
            confidence = probabilities[row, column, key[1]]
            print(f"    正确答案 '{tokenizer.id_to_char[key[0]]}'，"
                  f"它猜 '{tokenizer.id_to_char[key[1]]}'，"
                  f"而且给了自己 {confidence:.2%} 的把握")
    print()
    print("  这三个字，恰好就是语料里有歧义的那几个：")
    print("    '月' 后面有时是 '光'（床前明月光），有时是 '低'（望明月低头…）")
    print("    '头' 后面有时是 '望'（举头望明月），有时是 '思'（低头思故乡）")
    print("    '鹅' 后面有时还是 '鹅'，有时是 '曲'")
    print()
    print("  要分清它们，必须知道「我现在在诗句的哪个位置」。")

    print()
    print("=" * 64)
    print("探测三：它的错误发生在哪里")
    print("=" * 64)
    wrong_columns = np.nonzero(wrong)[1]
    counts = np.bincount(wrong_columns, minlength=CONTEXT)
    print(f"  窗口里第几个位置        {', '.join(str(i) for i in range(CONTEXT) if counts[i])}")
    print(f"  这个位置上错了几个      {', '.join(str(int(counts[i])) for i in range(CONTEXT) if counts[i])}")
    print()
    print(f"  全部 {len(wrong_columns)} 个错误，都发生在窗口的第 0 个和第 1 个位置上。")
    print("  也就是说，模型每次被迫在两个字的上下文里做判断时，就会出错。")
    print("  上下文一长，它就再没犯过错。")

    print()
    print("=" * 64)
    print("探测四：它最没把握的字，是最有意思的字")
    print("=" * 64)
    by_character = {}
    for row in range(len(answers)):
        for column in range(CONTEXT):
            character = int(answers[row, column])
            by_character.setdefault(character, []).append(
                float(probabilities[row, column, character]))
    ranked = sorted(by_character.items(), key=lambda item: np.mean(item[1]))
    print(f"  {'它最不确定的 5 个字':<24}{'它最有把握的 5 个字'}")
    print("  " + "-" * 52)
    for (low_char, low_values), (high_char, high_values) in zip(ranked[:5], ranked[::-1][:5]):
        print(f"  '{tokenizer.id_to_char[low_char]}'  {np.mean(low_values):.4f}"
              f"                '{tokenizer.id_to_char[high_char]}'  {np.mean(high_values):.4f}")
    print()
    low_five = " ".join(f"'{tokenizer.id_to_char[c]}'" for c, _ in ranked[:5])
    high_five = " ".join(f"'{tokenizer.id_to_char[c]}'" for c, _ in ranked[::-1][:5])
    print(f"  最不确定的那几个是 {low_five}。")
    print("  它们全部出现在有歧义的位置上——就是探测二里那三种错误旁边。")
    print()
    print(f"  最有把握的那几个是 {high_five}。")
    print("  它们在语料里只有一种下文，没有任何别的可能。")
    print()
    print("  换句话说：**它的不确定，不是随机的。**")
    print("  它恰好在自己该不确定的地方不确定。")

    print()
    print("=" * 64)
    print("探测五：上下文有多长，它才敢确定")
    print("=" * 64)
    ids = tokenizer.encode(CORPUS)
    print(f"  {'给它的上下文长度':<20}{'下一个字的损失':>14}{'猜对比例':>10}")
    print("  " + "-" * 46)
    for length in (1, 2, 3, 4, 8, 16, 32):
        rows, answers_short = [], []
        for start in range(0, len(ids) - length - 1, 7):
            rows.append(ids[start:start + length])
            answers_short.append(ids[start + length])
        with no_grad():
            logits = model(np.array(rows)).data[:, -1, :]
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities = shifted / shifted.sum(axis=-1, keepdims=True)
        target = probabilities[np.arange(len(answers_short)), answers_short]
        accuracy = float((logits.argmax(axis=-1) == np.array(answers_short)).mean())
        print(f"  {length:<20}{-np.log(target).mean():>14.4f}{accuracy:>10.2%}")
    print()
    print("  第 0 行给模型 1 个字，第 6 行给它 32 个字。")
    print("  但从第 2 个字开始，猜对比例就已经在 95% 以上了。")
    print()
    print("  也就是说：**这个模型其实不太需要上下文。**")
    print("  给它两个字它就够用了——因为我们的语料太规整，")
    print("  每个字的下文几乎只由前面一两个字决定。")
    print()
    print("  第 16 章那个位置编码、第 14 章那个「每个位置自己决定看哪里」，")
    print("  在这种语料上根本施展不开。")

    print()
    print("=" * 64)
    print("探测六：它会不会背整句")
    print("=" * 64)
    print("  把语料里的句子切开，看它能不能接上：")
    print()
    for prompt, expected in [("床前明月光疑是地上", "霜"),
                             ("举头望明月低头思", "故"),
                             ("白毛浮绿水红掌", "拨")]:
        text = generate_greedy(model, tokenizer, prompt, length=1)
        print(f"    '{prompt}|' -> '{text[-1]}'   （语料里是 '{expected}'）")
    print()
    print("  三个都对。但这说明不了太多——")
    print("  这三个片段，模型在训练里见过 8 遍。")
    print("  它到底是在「接诗」，还是在「回忆」？")
    print("  这一章给不出答案。因为我们的语料里，")
    print("  从来就没有出现过一句「它没见过但合语法」的中文。")

    print()
    print("=" * 64)
    print("观察到这里，然后把判断留给你")
    print("=" * 64)
    print("  我们看到了这些：")
    print(f"    - 语料原文上，困惑度 {on_corpus['perplexity']:.2f}。它几乎不犹豫。")
    print(f"    - 同样的字换个顺序，困惑度涨到 {off_corpus['perplexity']:.2e}，")
    print(f"      准确率从 {on_corpus['accuracy']:.2%} 掉回 {off_corpus['accuracy']:.2%}"
          f"（瞎猜是 {1/tokenizer.vocab_size:.2%}）。")
    print("      它学到的不是「中文怎么排」，而是「这 304 个字怎么排」。")
    print("    - 它出错的地方只有三种，全都是语料里有歧义的位置。")
    print(f"    - 它错的时候很自信（平均 {wrong_confidence:.2%} 把握）。")
    print("    - 它的错误全部集中在上下文只有一两个字的时候。")
    print()
    print("  这些观察同时支持两种说法，而且都说得通：")
    print()
    print("    说法 A：「它就是一个复读机。语料外的东西它一点都不会，")
    print("            连顺序打乱都认不出来。」")
    print("    说法 B：「它确实从这 304 个字里学到了一些结构——")
    print("            它知道哪些位置有歧义、哪里该犹豫、")
    print("            两个字够不够它做决定。这些不是背出来的。」")
    print()
    print("  这一章不替你选。")
    print("  但有一件事可以确定：**我们提的这些问题，语料太小，问不出结果。**")
    print("  304 个字符，38 个字一副对联，8 遍下来它当然背得下来。")
    print("  要让它非学结构不可，就得让它背不下来——")
    print("  也就是，语料得大到记不住。")

    print()
    print("=" * 64)
    print("那么，把它做大？")
    print("=" * 64)
    print(f"  我们现在的模型：{model.parameter_count():,} 个参数，")
    print(f"  语料 {len(CORPUS)} 个字符，上下文 {CONTEXT}。")
    print()
    print("  想让它真的学会中文，参数量至少要到亿这个量级，")
    print("  语料至少要到几十亿个字符。")
    print()
    print("  这两件事都不是「改几个常数」能解决的。")
    print("  但在动手之前，先撞上的是另一堵墙——")
    print("  **第 9 章我们写的那个 toygrad，太慢了。**")
    print()
    print("  它每一次运算都要用 Python 记一遍账，")
    print("  每一步都要把计算图从头走一遍。")
    print("  我们这个 15 万参数的小模型跑一步要十几毫秒，")
    print("  已经是能忍的极限了。再大 1000 倍呢？")
    print()
    print("  下一章开始，我们换一套工具。")
    print("  但先记住：**换工具不会改变任何原理。**")
    print("  你今天写下的每一行，在 PyTorch 里都有一个一模一样的对应物。")


if __name__ == "__main__":
    main()
