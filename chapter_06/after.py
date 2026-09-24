"""第 6 章的 after：不再一个一个试，而是把方向算出来。

这一章我们做三件事：

1. 用"挪一点点看损失怎么变"的办法，把每个参数该往哪边改算出来；
2. 一次把 32 个参数全部按方向挪一小步：w = w - 学习率 * 梯度；
3. 顺带手推一下我们这个模型的梯度公式，让训练快起来。

跑完这个文件，损失会从 0.6931 降到 0.01 以下，10 句话全部分对。
"""

import math

# ------------------------------------------------------------------ 第 1 章的东西：拆词
# 和第 1 章一模一样，一个字都没改。

VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]
UNKNOWN_WORD = "<UNK>"
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)

# ------------------------------------------------------------------ 第 2 章的东西：语料

CORPUS = [
    ("苹果发布新手机", "科技"),
    ("苹果发布新芯片", "科技"),
    ("华为发布新电脑", "科技"),
    ("小米发布新手机", "科技"),
    ("苹果芯片很强", "科技"),
    ("苹果很好吃", "食品"),
    ("苹果很甜", "食品"),
    ("香蕉很好吃", "食品"),
    ("这个苹果真甜", "食品"),
    ("苹果做成派", "食品"),
]

CATEGORIES = ["科技", "食品"]


def split_sentence(sentence):
    """把一句话拆成词。词表里没有的字换成一个占位符。"""
    words = []
    position = 0
    while position < len(sentence):
        matched_word = None
        for length in range(MAX_WORD_LENGTH, 0, -1):
            piece = sentence[position:position + length]
            if piece in WORD_TO_ID:
                matched_word = piece
                break
        if matched_word is None:
            words.append(UNKNOWN_WORD)
            position += 1
        else:
            words.append(matched_word)
            position += len(matched_word)
    return words


# ------------------------------------------------------------------ 第 3 到第 5 章的东西：分数、概率、损失
# 这三段也一个字都没改。这一章要改的只有一件事：参数怎么更新。

# 全书的参数表：每个词有两个数，一个是"它有多像科技词"，一个是"它有多像食品词"。
# 16 个词 × 2 个类别 = 32 个参数。
WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}


def score(words):
    """一句话的分数：把这个句子里每个词的权重分别加起来。"""
    scores = [0.0, 0.0]
    for word in words:
        if word not in WEIGHT_TABLE:
            continue
        weights = WEIGHT_TABLE[word]
        scores[0] += weights[0]
        scores[1] += weights[1]
    return scores


def softmax(scores):
    """把两个分数变成两个加起来等于 1 的概率。"""
    biggest = max(scores)
    exponentials = [math.exp(value - biggest) for value in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def loss_of(words, label):
    """一句话的损失：正确答案的概率越低，损失越大。"""
    return -math.log(softmax(score(words))[CATEGORIES.index(label)])


def average_loss(corpus):
    total = 0.0
    for sentence, label in corpus:
        total += loss_of(split_sentence(sentence), label)
    return total / len(corpus)


def accuracy(corpus):
    right = 0
    for sentence, label in corpus:
        scores = score(split_sentence(sentence))
        answer = CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]
        if answer == label:
            right += 1
    return right


# ------------------------------------------------------------------ 本章主角：梯度


def numeric_gradient(table, corpus, h=1e-3):
    """数值法求梯度：把每个参数单独挪一点点，看损失怎么变。

    用的是中心差分：

        (损失(参数 + h) - 损失(参数 - h)) / (2h)

    正负各挪一次再相减，两边的一阶误差互相抵消，比只挪一边准得多。
    这个办法不需要任何推导，模型换成别的样子它也照样能用。

    返回值是一个和 table 形状一样的字典：
    result[词][类别] 就是"把这个参数往正方向挪一点点，损失会怎么变"。
    """
    result = {word: [0.0, 0.0] for word in VOCABULARY}

    for word in VOCABULARY:
        for category_index in (0, 1):
            original = table[word][category_index]

            table[word][category_index] = original + h
            plus = average_loss(corpus)
            table[word][category_index] = original - h
            minus = average_loss(corpus)
            table[word][category_index] = original

            result[word][category_index] = (plus - minus) / (2 * h)

    return result


def gradient(table, corpus):
    """手推出来的梯度公式。

    对"把词权重加起来 -> softmax -> 交叉熵"这一串，推到最后结果是：

        每个样本贡献的梯度 = (预测概率 - 正确答案) × 这个词出现过几次

    也就是 (p - y)。预测概率比正确答案高，就说明这一类被高估了，
    这个句子里每个词的权重都要往下压一点。
    """
    result = {word: [0.0, 0.0] for word in VOCABULARY}
    n = len(corpus)

    for sentence, label in corpus:
        words = split_sentence(sentence)
        probabilities = softmax(score(words))
        correct_index = CATEGORIES.index(label)

        for category_index in (0, 1):
            difference = (probabilities[category_index]
                          - (1.0 if category_index == correct_index else 0.0))
            difference /= n
            for word in words:
                if word in result:
                    result[word][category_index] += difference

    return result


# ------------------------------------------------------------------ 训练
# 这一章真正多出来的只有下面这一行：
#
#     w = w - 学习率 * 梯度
#
# 梯度告诉我们"往哪边、有多陡"，学习率告诉我们"走多远"。


def train(steps=200, lr=0.5):
    """从全 0 出发，跑 steps 轮梯度下降。返回每一步的损失。"""
    history = []

    for _ in range(steps):
        grads = gradient(WEIGHT_TABLE, CORPUS)
        for word in VOCABULARY:
            for category_index in (0, 1):
                WEIGHT_TABLE[word][category_index] -= (
                    lr * grads[word][category_index]
                )
        history.append(average_loss(CORPUS))

    return history


def reset_table():
    """把参数表清回全 0，方便反复跑。"""
    for word in VOCABULARY:
        WEIGHT_TABLE[word] = [0.0, 0.0]


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


# ------------------------------------------------------------------ main


def main():
    print("=" * 66)
    print("用算出来的梯度训练")
    print("=" * 66)
    print("更新规则：w = w - 学习率 × 梯度")
    print()

    reset_table()
    steps = 200
    lr = 0.5
    history = train(steps=steps, lr=lr)

    print(f"  学习率 = {lr}，一共 {steps} 轮")
    print(f"  轮次   损失")
    for index in range(0, steps, 20):
        print(f"   {index + 1:>3}    {history[index]:.8f}")
    print()
    print(f"  训练结束：损失 = {history[-1]:.8f}，正确率 = {accuracy(CORPUS)}/10")
    print()

    print("=" * 66)
    print("它学到的参数表")
    print("=" * 66)
    print(f"  {pad('词', 8)}{'科技':>8}{'食品':>8}"
          f"   出现在几句科技句 / 几句食品句")
    for word in VOCABULARY:
        technology, food = WEIGHT_TABLE[word]
        in_tech = sum(1 for sentence, label in CORPUS
                      if label == "科技" and word in split_sentence(sentence))
        in_food = sum(1 for sentence, label in CORPUS
                      if label == "食品" and word in split_sentence(sentence))
        print(f"  {pad(word, 8)}{technology:>8.3f}{food:>8.3f}"
              f"        {in_tech} / {in_food}")
    print()
    print("  读这张表的方法：看最后一列。")
    print("  「发布」4 句科技、0 句食品 —— 权重最大（1.035）。")
    print("  「电脑」1 句科技、0 句食品 —— 方向对，但只见过一次，权重小（0.267）。")
    print("  「苹果」3 句科技、4 句食品 —— 两边几乎抵消，权重被拉平（-0.342）。")
    print("  一个词的权重，取决于它在两边的句子里各出现过多少次。")
    print()

    print("=" * 66)
    print("逐句检查")
    print("=" * 66)
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        probabilities = softmax(score(words))
        correct_index = CATEGORIES.index(label)
        answer = CATEGORIES[0] if probabilities[0] >= probabilities[1] else CATEGORIES[1]
        mark = "对" if answer == label else "错"
        print(f"  {mark}  {pad(sentence, 18)}判成 {answer}，"
              f"正确答案的概率 {probabilities[correct_index]:.4f}")
    print()
    print("到这里，我们已经把「学习」这件事整个写完了：")
    print("前向算分 -> 用损失衡量错得多离谱 -> 用梯度决定往哪边改 -> 改。")
    print()
    print("剩下的问题是：梯度的公式是我们人推的。")
    print("这个模型只有一层，推一次就够。层数一多呢？")


if __name__ == "__main__":
    main()
