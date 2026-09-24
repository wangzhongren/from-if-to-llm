"""第 6 章的 before：第 5 章那套"试着改"的办法，走到头看看。

第 5 章我们有了损失 —— 一个数字，衡量这次分得有多离谱。
但损失是个总分：它不会告诉我们 32 个参数里，
到底是哪一个该往哪边挪、挪多少。

第 5 章用的办法是挨个试：把某个参数加一点，重新算一遍损失，
变小了就留下，变大了就试减一点。

这个文件把这件事做完。它确实能用，也确实在变好 —— 只是慢到没法用。
"""

import math

# ------------------------------------------------------------------ 前五章的东西，照搬

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

WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}

# 每试一次、每改一次，我们就把整份语料重算一遍。这个计数器记的就是这个次数。
FORWARD_COUNT = 0


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
    biggest = max(scores)
    exponentials = [math.exp(value - biggest) for value in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def loss_of(words, label):
    return -math.log(softmax(score(words))[CATEGORIES.index(label)])


def average_loss(corpus):
    """把整份语料算一遍。这就是我们数的"一次前向"。"""
    global FORWARD_COUNT
    FORWARD_COUNT += 1
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


# ------------------------------------------------------------------ 第 5 章的改法


def try_adjust_one_parameter(word, category_index, amount, current_loss):
    """加一点试试，减一点试试，哪边让损失变小就留哪边。"""
    original = WEIGHT_TABLE[word][category_index]
    best_value = original
    best_loss = current_loss

    for delta in (amount, -amount):
        WEIGHT_TABLE[word][category_index] = original + delta
        candidate = average_loss(CORPUS)
        if candidate < best_loss - 1e-12:
            best_loss = candidate
            best_value = original + delta

    WEIGHT_TABLE[word][category_index] = best_value
    return best_loss


def adjust_weights_once(amount):
    """把所有 32 个参数扫一遍，一个个试。"""
    current = average_loss(CORPUS)
    for word in VOCABULARY:
        for category_index in (0, 1):
            current = try_adjust_one_parameter(word, category_index, amount,
                                               current)
    return current


def reset_table():
    for word in VOCABULARY:
        WEIGHT_TABLE[word] = [0.0, 0.0]


# ------------------------------------------------------------------ main


def rounds_to_target(amount, target=0.001, max_rounds=200):
    """一直改，改到损失小于 target 为止。返回（用了几轮，损失是多少）。"""
    reset_table()
    current = average_loss(CORPUS)
    for round_number in range(1, max_rounds + 1):
        current = adjust_weights_once(amount)
        if current < target:
            return round_number, current
    return None, current


def main():
    global FORWARD_COUNT

    print("=" * 66)
    print("第 5 章那套办法：一个参数一个参数地试着改")
    print("=" * 66)
    print("规则：把某个参数加一点点，重新算一遍损失。")
    print("      变小了就留下，变大了就试试减一点点。")
    print(f"参数一共 {len(VOCABULARY) * 2} 个（16 个词 × 2 个类别）。")
    print(f"每次扫一遍所有参数，要跑 {len(VOCABULARY) * 2 * 2} 次完整的前向计算")
    print("（每个参数加一次、减一次，各算一遍损失）。")
    print()

    print("  我们只改一个东西：每次挪多少。")
    print()
    print(f"  {'每次挪多少':>10}{'达到损失 < 0.001 要几轮':>24}"
          f"{'用掉的前向次数':>16}")
    for amount in (0.005, 0.01, 0.05, 0.5, 5.0):
        FORWARD_COUNT = 0
        rounds, final = rounds_to_target(amount)
        if rounds is None:
            text = f"200 轮还没到（{final:.4f}）"
            used = f"{FORWARD_COUNT}"
        else:
            text = f"{rounds}"
            used = f"{FORWARD_COUNT}"
        print(f"  {amount:>10}{text:>24}{used:>16}")

    print()
    print("  同一个办法，同一个模型，只因为「每次挪多少」不同：")
    print("  挪得太小，200 轮都到不了；挪得够大，1 轮就搞定。")
    print("  差了几百倍的工作量 —— 而这个数填多少，事先没有任何依据。")
    print()

    print("=" * 66)
    print("问题在哪")
    print("=" * 66)
    print("一、这个「每次挪多少」是靠猜的。上面那张表，我们是先跑了才知道。")
    print()
    print("二、它一次只能动一个参数。别的 31 个参数只能干等着 ——")
    print("    可它们之间明明是有关系的（「手机」和「新」要一起使劲，")
    print("    才能把「苹果发布新手机」这种句子推到科技那一边）。")
    print()
    print("    那让 32 个参数一起动呢？每个参数有「加」「减」两个选择，")
    print("    合起来是 2^32 ≈ 43 亿种组合。一个一个试过去，试不完。")
    print()
    print("三、它要跑的次数是「参数个数的两倍」。")
    print(f"    我们这里只有 {len(VOCABULARY) * 2} 个参数，一轮 {len(VOCABULARY) * 2 * 2} 次，还撑得住。")
    print("    要是参数有一百万个呢？一轮就是两百万次。")
    print("    今天最大的那些模型，有几千亿个参数。这条路走不通。")
    print()
    print("三条其实是同一件事：")
    print("  我们现在只会问「这次分错了」，不会问「每个参数该挪多少」。")
    print()
    print("这就是第 5 章留下的问题：")
    print("  能不能一次把「每个参数该往哪边改、改多少」全部算出来？")


if __name__ == "__main__":
    main()
