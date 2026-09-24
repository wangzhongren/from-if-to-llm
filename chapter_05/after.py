"""
这个文件演示本章的产物：一把能说出"错得有多离谱"的尺子。

两件事：
1. 把分数变成概率（正确答案有多大概率）——用的是 softmax；
2. 用"正确答案的概率"算出一个数：概率越低，这个数越大。
   它的名字叫损失（loss），这里用的是交叉熵。

有了这把尺子，我们就能做第 4 章做不到的事：
    把某个权重改一点点，看损失是变大了还是变小了。
第 4 章的尺子只会说"对"或"错"，改完还是"对"或"错"；
这把尺子是一个连续的、没有上限的数。

跑起来会看到一件很有意思的事：第 1 轮正确率就冲到 10/10 了，
而损失又继续降了 25 轮。正确率到顶之后，损失还能接着说话。
"""

import math

VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派",
]
UNKNOWN_WORD = "<UNK>"
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)

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

# 每次试着改动多少。这一章我们会看到：这个数第一次真的有意义了。
ADJUST_AMOUNT = 0.5


def split_sentence(sentence):
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


def display_width(text):
    """算一行字在终端里占几格（汉字占两格，ASCII 占一格），只为了表格对齐。"""
    return sum(2 if ord(character) > 127 else 1 for character in text)


def pad(text, width):
    """把 text 补齐到 width 格宽。"""
    return text + " " * max(0, width - display_width(text))


def format_loss(value):
    """
    损失会越变越小，小到小数点后四位看不出来，所以很小的数改用科学计数法。

    取绝对值是为了对付浮点误差：概率算出来可能是 1.0000000000000002，
    于是 -log 会给出一个极小的负数，写成科学计数法就成了 -1.23e-16。
    """
    if value < 0.0001:
        return f"{abs(value):.2e}"
    return f"{value:.4f}"


# ---------- 起点：和第 4 章一样，一张全 0 的表 ----------

WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}


# ---------- 本章的新东西：从分数到概率，从概率到损失 ----------

def score(words):
    scores = [0.0, 0.0]
    for word in words:
        if word not in WEIGHT_TABLE:
            continue
        weights = WEIGHT_TABLE[word]
        scores[0] += weights[0]
        scores[1] += weights[1]
    return scores


def softmax(scores):
    """
    把一组分数变成一组概率：每个都在 0 到 1 之间，加起来正好是 1。

    做法：先把每个分数换成 e 的那么多次方，再除以它们的总和。
    先减去最大的那个分数，是为了别让 e 的指数太大撑爆浮点数——
    这一步只是同时缩小所有分子分母，不改变结果。
    """
    biggest = max(scores)
    exponentials = [math.exp(value - biggest) for value in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def probability_of(words, category_index):
    return softmax(score(words))[category_index]


def loss_of(words, label):
    """
    一句话的损失：正确答案的概率越低，损失越大。

    公式是 -log(正确答案的概率)：概率是 1 时损失为 0，
    概率掉到 0.5 时损失约 0.69，概率掉到 0.01 时损失约 4.6。
    """
    correct_index = CATEGORIES.index(label)
    return -math.log(probability_of(words, correct_index))


def average_loss(corpus):
    total = 0.0
    for sentence, label in corpus:
        total += loss_of(split_sentence(sentence), label)
    return total / len(corpus)


# ---------- 有了尺子，就能试着改 ----------

def try_adjust_one_parameter(corpus, word, category_index, amount, current_loss):
    """
    试着把一个参数"加一点"，算一遍损失；再试着"减一点"，算一遍损失。
    哪边让损失变小就留下哪个，两边都变差就退回原样。
    返回改完之后的损失。
    """
    original_value = WEIGHT_TABLE[word][category_index]
    best_value = original_value
    best_loss = current_loss
    for delta in (amount, -amount):
        WEIGHT_TABLE[word][category_index] = original_value + delta
        candidate_loss = average_loss(corpus)
        if candidate_loss < best_loss - 1e-12:
            best_loss = candidate_loss
            best_value = original_value + delta
    WEIGHT_TABLE[word][category_index] = best_value
    return best_loss


def adjust_weights_once(corpus, amount):
    """
    把所有参数扫一遍，一个个试。

    这是最笨的改法：一次只动一个数，动完还要把整个语料重新算一遍。
    """
    loss = average_loss(corpus)
    changed_count = 0
    for word in VOCABULARY:
        for category_index in (0, 1):
            original_value = WEIGHT_TABLE[word][category_index]
            loss = try_adjust_one_parameter(corpus, word, category_index, amount, loss)
            if WEIGHT_TABLE[word][category_index] != original_value:
                changed_count += 1
    return loss, changed_count


def accuracy(corpus):
    right = 0
    for sentence, label in corpus:
        words = split_sentence(sentence)
        scores = score(words)
        answer = CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]
        if answer == label:
            right += 1
    return right


def main():
    print("=" * 66)
    print("起点：全 0 的表，和第 4 章一样")
    print("=" * 66)
    print(f"  正确率：{accuracy(CORPUS)}/10")
    print(f"  损失：  {format_loss(average_loss(CORPUS))}")
    print()
    print("  逐句看（概率 = 正确答案的概率）：")
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        correct_index = CATEGORIES.index(label)
        probability = probability_of(words, correct_index)
        print(f"    {pad(sentence, 18)} 概率 {probability:.3f}   损失 {format_loss(-math.log(probability))}")
    print()
    print("  每一句的概率都是 0.500，每一句的损失都是 0.6931。")
    print("  0.6931 就是 -log(0.5)：两个类别各占一半，等于什么都不知道。")
    print("  记住这个起点，它就是「完全不知道」的样子。")
    print()

    print("=" * 66)
    print(f"用损失当尺子，每次改 {ADJUST_AMOUNT}")
    print("=" * 66)
    print("  轮次   损失        这一轮改了几个参数   正确率")
    for round_number in range(1, 31):
        best_loss, changed_count = adjust_weights_once(CORPUS, amount=ADJUST_AMOUNT)
        print(f"   {round_number:>2}    {format_loss(best_loss):<10}   {changed_count:>15}"
              f"   {accuracy(CORPUS)}/10")
        if changed_count == 0:
            break
    print()

    print("=" * 66)
    print("改进之后的表")
    print("=" * 66)
    for word, weights in WEIGHT_TABLE.items():
        print(f"  {pad(word, 8)} 科技 {weights[0]:>5.1f}   食品 {weights[1]:>5.1f}")
    print()

    print("=" * 66)
    print("再看每一句")
    print("=" * 66)
    for sentence, label in CORPUS:
        words = split_sentence(sentence)
        correct_index = CATEGORIES.index(label)
        probability = probability_of(words, correct_index)
        print(f"  {pad(sentence, 18)} 概率 {probability:.6f}   损失 {format_loss(-math.log(probability))}")
    print()
    print("-" * 66)
    print(f"正确率：5/10 -> 10/10，第 1 轮就到顶了，之后再也没动过。")
    print(f"损失：  0.6931 -> {format_loss(average_loss(CORPUS))}，一路降下来，没有到顶这一说。")
    print()
    print("两把尺子的区别就在这里：")
    print("    '答错了几句'只有 11 个刻度，答完 10 句就到头了；")
    print("    '损失'是一个连续的数，没有上限也（几乎）没有下限。")
    print("    所以它能一直告诉你：离'完全确定'还有多远。")
    print()
    print("最后看一件小事：权重涨到了 12.5。")
    print("损失想让每一句的概率都趋近于 1，于是它一直奖励'更笃定'，")
    print("权重就一路涨下去。什么时候停？")
    print("我们现在的规矩是'再没有任何一个参数能改进损失了'——")
    print("这个规矩是我们定的，不是损失定的。")


if __name__ == "__main__":
    main()
