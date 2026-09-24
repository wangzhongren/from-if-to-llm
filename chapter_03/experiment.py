"""
这个文件让读者亲眼看到"数票"这件事有多依赖我们手动做的决定。

第 2 章的规则要一条条写，太累。数票法看上去是个进步：
每个词投一票，票多的赢，规则不用写了。

但"每个词投哪一边"这件事还是要人定。这个文件摆出三种摆法：
    A. 按多数派自动归边（before.py 的做法）
    B. 按多数派，但把"苹果"改成科技
    C. 两边都出现过的词弃权，不投票
三种摆法，两种结果。然后我们再看加权分数是怎么绕开这个问题的。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 别的章节也可能有同名的 before.py / after.py，先清掉缓存
sys.modules.pop("before", None)
sys.modules.pop("after", None)

from before import CORPUS, VOCABULARY, count_appearances, pad, split_sentence
from after import COEFFICIENT_TABLE, judge as judge_by_score, score


def find_ambiguous_words(corpus):
    """找一找：哪些词在两个类别里都出现过？（这些词最难办）"""
    technology_count, food_count = count_appearances(corpus)
    return [word for word in VOCABULARY
            if technology_count[word] > 0 and food_count[word] > 0]


def build_table(corpus, override=None):
    """按多数派给每个词定一个类别；override 里的词强行指定。"""
    technology_count, food_count = count_appearances(corpus)
    table = {}
    for word in VOCABULARY:
        if technology_count[word] > food_count[word]:
            table[word] = "科技"
        else:
            table[word] = "食品"
    if override:
        table.update(override)
    return table


def judge_by_votes(words, word_to_category, silent_words=frozenset()):
    """数票。silent_words 里的词不投票。"""
    technology_votes = 0
    food_votes = 0
    for word in words:
        if word in silent_words:
            continue
        category = word_to_category.get(word)
        if category == "科技":
            technology_votes += 1
        elif category == "食品":
            food_votes += 1
    if technology_votes == food_votes:
        return f"平局({technology_votes}:{food_votes})"
    return "科技" if technology_votes > food_votes else "食品"


def measure(judge_function):
    """跑一遍语料，返回答对的句数和答错的句子。"""
    right = 0
    wrong = []
    for sentence, label in CORPUS:
        answer = judge_function(split_sentence(sentence))
        if answer == label:
            right += 1
        else:
            wrong.append((sentence, answer, label))
    return right, wrong


def main():
    ambiguous = find_ambiguous_words(CORPUS)
    print("=" * 66)
    print("先找一找：哪些词在两个类别里都出现过？")
    print("=" * 66)
    print(f"  {('、'.join(ambiguous))}")
    print("  这些词没法'归到某一边'，而每个词恰恰只能写一个类别。")
    print()

    print("=" * 66)
    print("三种摆法")
    print("=" * 66)

    table_a = build_table(CORPUS)
    table_b = build_table(CORPUS, override={"苹果": "科技"})
    table_c = table_a

    variants = [
        ("A. 按多数派自动归边", lambda words: judge_by_votes(words, table_a)),
        ("B. 把'苹果'改成科技", lambda words: judge_by_votes(words, table_b)),
        ("C. 两边都有的词弃权", lambda words: judge_by_votes(words, table_c, silent_words=frozenset(ambiguous))),
    ]
    for name, judge_function in variants:
        right, wrong = measure(judge_function)
        detail = ""
        if wrong:
            sentence, answer, label = wrong[0]
            detail = f"   错的是「{sentence}」，答成了{answer}"
        print(f"  {pad(name, 24)} 正确 {right:>2}/10{detail}")
    print()
    print("  摆法 A 和 B 的区别，只是'苹果'这个词放在哪一边。")
    print("  一个词的位置，决定了整体对错——而这个位置是我们随手定的。")
    print("  摆法 C 也全对了，但它是靠'把两个词从票箱里拿出来'做到的：")
    print("  它等于承认了这两个词没法归类，只好不让它们说话。")
    print()

    print("=" * 66)
    print("加权分数：一个词不需要选边")
    print("=" * 66)
    right, wrong = measure(lambda words: judge_by_score(words))
    print(f"  加权分数               正确 {right:>2}/10")
    print()
    print("  关键在系数表里的这一行：")
    print(f"    '苹果' -> 科技 +{COEFFICIENT_TABLE['苹果'][0]}   食品 +{COEFFICIENT_TABLE['苹果'][1]}")
    print("  它同时给两边分量，食品那边多一点，科技那边少一点。")
    print("  不用做「这个词归谁」的决定，也就没有摆法 A/B/C 的差别。")
    print()

    print("=" * 66)
    print("'分量'这件事，票数根本表达不了")
    print("=" * 66)
    print("  在数票法里，每个词的分量都是 1：")
    print("    '芯片' 1 票   '电脑' 1 票   '新' 1 票   '派' 1 票")
    print("  '苹果芯片很强'这句里，'芯片'顶了 1 票；")
    print("  可它明显比'新'这种词更能说明问题。票数说不出这件事。")
    print()
    print("  换成系数之后：")
    for word in ["芯片", "电脑", "新", "手机"]:
        technology_coefficient, food_coefficient = COEFFICIENT_TABLE[word]
        print(f"    {pad(word, 8)} 科技 +{technology_coefficient}   食品 +{food_coefficient}")
    print("  每个词的分量可以不一样，都是我们自己填的。")
    print()
    print("-" * 66)
    print("数票法不是不能用。它的问题是：每一次'这个词算哪边'，都要我们手动拍板。")
    print("加权分数把'拍板'变成了'填数字'——而数字，是程序可以自己改的东西。")


if __name__ == "__main__":
    main()
