**English** | [中文](README.md)

# Chapter 3: Turning if into numbers

## The problem in this chapter

The last chapter left us with a question:

> Tens of thousands of words — are we supposed to write tens of thousands of ifs?

That is not an exaggeration. In Chapter 2 we hand-patched 8 rules to cover **16 words**. At
that ratio, a 50,000-word vocabulary would need more than twenty thousand rules — and
experiment 2 already told us that **not one of those rules can be in the wrong order.**

So what this chapter is looking for isn't "better rules". It's **a path that doesn't require
writing rules by hand.**

First, look at what the Chapter 2 rules are actually doing. Line them up next to each other and
you'll notice they all have the same shape:

```python
if "手机" in words:
    return "科技"
if "芯片" in words:
    return "科技"
if "好吃" in words:
    return "食品"
```

Every rule is saying the same thing: **when this word appears in the sentence, count it as
evidence for one of the categories.** The only thing that differs between one rule and another
is "which category does this word count for".

So let's stop writing ifs. Let's make a table:

```text
手机 -> 科技
芯片 -> 科技
好吃 -> 食品
```

*(手机, 芯片 -> 科技 ("technology"); 好吃 -> 食品 ("food").)*

A new sentence comes in: look each of its words up in the table, and see which category collects
more words.

## The simplest attempt

We don't have to fill this table in either. **Let the program count for itself**: for each word,
whichever category it appears in more often in the corpus is the category we put it in.

```python
def build_word_to_category(corpus):
    technology_count, food_count = count_appearances(corpus)
    table = {}
    for word in VOCABULARY:
        if technology_count[word] > food_count[word]:
            table[word] = "科技"
        else:
            table[word] = "食品"
    return table
```

Run `before.py`, and this is the table the program counts up:

```text
  苹果     科技 3 句 / 食品 4 句   ->  食品
  发布     科技 4 句 / 食品 0 句   ->  科技
  新       科技 4 句 / 食品 0 句   ->  科技
  手机     科技 2 句 / 食品 0 句   ->  科技
  芯片     科技 2 句 / 食品 0 句   ->  科技
  电脑     科技 1 句 / 食品 0 句   ->  科技
  华为     科技 1 句 / 食品 0 句   ->  科技
  小米     科技 1 句 / 食品 0 句   ->  科技
  好吃     科技 0 句 / 食品 2 句   ->  食品
  很       科技 1 句 / 食品 3 句   ->  食品
  甜       科技 0 句 / 食品 2 句   ->  食品
  香蕉     科技 0 句 / 食品 1 句   ->  食品
  这个     科技 0 句 / 食品 1 句   ->  食品
  真       科技 0 句 / 食品 1 句   ->  食品
  做成     科技 0 句 / 食品 1 句   ->  食品
  派       科技 0 句 / 食品 1 句   ->  食品
```

*(Each row: a word, its technology count, its food count, and the category it gets assigned.
苹果 goes to food (4 sentences vs 3), 很 goes to food (3 vs 1).)*

Now count the votes:

```text
  [对] 苹果发布新手机     科技 3 票 / 食品 1 票  答：科技   真实：科技
  [对] 苹果发布新芯片     科技 3 票 / 食品 1 票  答：科技   真实：科技
  [对] 华为发布新电脑     科技 4 票 / 食品 0 票  答：科技   真实：科技
  [对] 小米发布新手机     科技 4 票 / 食品 0 票  答：科技   真实：科技
  [错] 苹果芯片很强       科技 1 票 / 食品 2 票  答：食品   真实：科技
  [对] 苹果很好吃         科技 0 票 / 食品 3 票  答：食品   真实：食品
  [对] 苹果很甜           科技 0 票 / 食品 3 票  答：食品   真实：食品
  [对] 香蕉很好吃         科技 0 票 / 食品 3 票  答：食品   真实：食品
  [对] 这个苹果真甜       科技 0 票 / 食品 4 票  答：食品   真实：食品
  [对] 苹果做成派         科技 0 票 / 食品 3 票  答：食品   真实：食品

------------------------------------------------------------------
正确：9/10
```

*(Each row: a sentence, its technology votes, its food votes, the answer, the true label.
The last line of the block: `正确：9/10` — "correct: 9/10".)*

9/10. The one it gets wrong is exactly the sentence we deliberately left a catch-all rule for in
Chapter 2.

The reason is in the vote counts:

- 苹果 is assigned to food (food 4 sentences, technology 3);
- 很 is also assigned to food (food 3 sentences, technology 1);
- that leaves only 芯片 casting a vote for technology.

So 「苹果芯片很强」 ends up with **technology 1 vote, food 2 votes**, and the answer comes out
food.

The problem isn't that 苹果 was assigned to the wrong side. The problem is: **every word can only
pick one side, and 苹果 was always on both sides to begin with.**

## Experiment

Run `experiment.py`. We lay out three ways of "counting votes".

```text
==================================================================
先找一找：哪些词在两个类别里都出现过？
==================================================================
  苹果、很
  这些词没法'归到某一边'，而每个词恰恰只能写一个类别。

==================================================================
三种摆法
==================================================================
  A. 按多数派自动归边      正确  9/10   错的是「苹果芯片很强」，答成了食品
  B. 把'苹果'改成科技      正确 10/10
  C. 两边都有的词弃权      正确 10/10

  摆法 A 和 B 的区别，只是'苹果'这个词放在哪一边。
  一个词的位置，决定了整体对错——而这个位置是我们随手定的。
  摆法 C 也全对了，但它是靠'把两个词从票箱里拿出来'做到的：
  它等于承认了这两个词没法归类，只好不让它们说话。

==================================================================
加权分数：一个词不需要选边
==================================================================
  加权分数               正确 10/10

  关键在系数表里的这一行：
    '苹果' -> 科技 +0.2   食品 +0.9
  它同时给两边分量，食品那边多一点，科技那边少一点。
  不用做「这个词归谁」的决定，也就没有摆法 A/B/C 的差别。
```

*(First, the words that appear in both categories: 苹果 and 很. Those words can't be assigned to
one side, yet each word can only be written into one category.*

*Then the three arrangements: A, assigning each word to its majority side, scores 9/10 and gets
「苹果芯片很强」 wrong (answered food). B, changing 苹果 to technology, scores 10/10. C, letting
words that appear on both sides abstain, also scores 10/10. The note underneath says the only
difference between A and B is which side 苹果 goes on — one word's position decides whether the
whole thing is right, and we picked that position at random. C is also all correct, but it got
there by pulling two words out of the ballot box: it amounts to admitting those two words can't
be categorized and silencing them.*

*Last section: "the coefficient sum — a word doesn't have to pick a side", 10/10. The line that
matters: 苹果 -> technology +0.2, food +0.9 — it gives both sides a share at once, more to food,
less to technology. No decision about "which side does this word belong to" means no A/B/C
distinction.)*

And here's something the vote counts can't express:

```text
  在数票法里，每个词的分量都是 1：
    '芯片' 1 票   '电脑' 1 票   '新' 1 票   '派' 1 票
  '苹果芯片很强'这句里，'芯片'顶了 1 票；
  可它明显比'新'这种词更能说明问题。票数说不出这件事。

  换成系数之后：
    芯片     科技 +1.5   食品 +0.0
    电脑     科技 +1.0   食品 +0.0
    新       科技 +0.5   食品 +0.0
    手机     科技 +1.2   食品 +0.0
  每个词的分量可以不一样，都是我们自己填的。
```

*(In vote counting every word has a share of exactly 1: 芯片 1 vote, 电脑 1 vote, 新 1 vote, 派
1 vote. In 「苹果芯片很强」, 芯片 is worth 1 vote — but it obviously says more about the sentence
than a word like 新 does. Votes can't say that.*

*With coefficients: 芯片 technology +1.5 / food +0.0, 电脑 +1.0 / +0.0, 新 +0.5 / +0.0, 手机
+1.2 / +0.0. Every word can have a different share, and we fill them all in ourselves.)*

## The new mechanism

This chapter's shift is one sentence long:

> **Don't assign words to categories. Record a share for each word.**

Concretely, every word is followed by two numbers:

```text
        这个词每出现一次，给"科技"加多少分，给"食品"加多少分
苹果  ->  (0.2, 0.9)
芯片  ->  (1.5, 0.0)
很    ->  (0.0, 0.3)
```

*(How many points to add to "technology" and how many to "food" each time this word appears:
苹果 -> (0.2, 0.9), 芯片 -> (1.5, 0.0), 很 -> (0.0, 0.3).)*

A sentence's score is just its words added up one by one:

```text
苹果发布新手机 = 苹果 + 发布 + 新 + 手机
科技分数 = 0.2 + 0.8 + 0.5 + 1.2 = 2.7
食品分数 = 0.9 + 0.0 + 0.0 + 0.0 = 0.9
```

*(苹果发布新手机 = 苹果 + 发布 + 新 + 手机; technology score = 0.2 + 0.8 + 0.5 + 1.2 = 2.7;
food score = 0.9 + 0.0 + 0.0 + 0.0 = 0.9.)*

Compare the two scores and the higher one wins. This number, we'll call it the **score**.

The number each word gets multiplied by, we'll call a **coefficient**. Every row of the
coefficient table is a pair of coefficients — one for the "technology" side, one for the "food"
side.

These words that the sentence is split into (along with how many times each one appears) are the
sentence's **features**. From here on we'll keep using this word: **the score is the sum of
feature times coefficient.**

It solves two things:

1. **苹果 doesn't have to pick a side any more.** 0.2 to technology, 0.9 to food — one word is
   evidence for both sides at once. The difference between arrangements A, B and C in the
   experiment simply doesn't exist here.
2. **The shares can differ.** 芯片 gives 1.5, 新 gives 0.5. Votes can't do that — in vote
   counting every word's vote is 1.

## Python implementation

The full code is in `after.py`, and the core of it is two things: one table and one sum.

```python
COEFFICIENT_TABLE = {
    "苹果": (0.2, 0.9),     # give both sides a share, but more to food
    "发布": (0.8, 0.0),
    "新":   (0.5, 0.0),
    "手机": (1.2, 0.0),
    # ...
}
```

```python
def score(words):
    technology_score = 0.0
    food_score = 0.0
    for word in words:
        if word not in COEFFICIENT_TABLE:
            continue
        technology_coefficient, food_coefficient = COEFFICIENT_TABLE[word]
        technology_score += technology_coefficient
        food_score += food_coefficient
    return technology_score, food_score
```

Three things to notice:

- Words that aren't in `COEFFICIENT_TABLE` (like `<UNK>`) are skipped outright. They have no
  place in the table, so they can't vote — which is the same as Chapter 2's "the rules can't see
  words outside the vocabulary".
- Every appearance adds once. So if 苹果 appears twice in one sentence, both coefficients get
  added twice.
- The comparison inside `judge` uses `>=`: when the scores are equal, "technology" counts first.
  A tie is abnormal to begin with (it means the sentence has no lean at all); for now we just
  pick one, and in Chapter 5 we'll come back and work out just how unreliable that is.

## What it solves

| | Vote counting (`before.py`) | Coefficients (`after.py`) |
|---|---|---|
| accuracy | 9/10 | 10/10 |
| what each word does | 1 vote | two coefficients, of different sizes |
| what happens to 苹果 | it must pick a side | it gives a share to both sides |

All 10 correct, and the process is visible:

```text
  [对] 苹果发布新手机     科技  2.7 / 食品  0.9   答：科技   真实：科技
  [对] 苹果芯片很强       科技  1.7 / 食品  1.2   答：科技   真实：科技
```

*(苹果发布新手机: technology 2.7 / food 0.9, answered technology, truly technology.
苹果芯片很强: technology 1.7 / food 1.2, answered technology, truly technology.)*

What you get beyond "right or wrong" is **by how much**:

- 「苹果发布新手机」 wins by 1.8 points — comfortable;
- 「苹果芯片很强」 wins by 0.5 points — barely.

Vote counting has counts too, but they're whole numbers, and one vote is indistinguishable from
another vote. Scores are continuous numbers: they can be big, or they can win by a hair.

**But one thing has to be said clearly: the 32 numbers in this table were typed in by hand, one
at a time.**

They weren't computed by the program and they didn't grow out of the data — we looked at these
10 sentences and filled them in by trial and error. What we were thinking as we filled them in
was "芯片 looks pretty important, let's give it 1.5". That is the real price of this chapter.

## What it still can't solve

Go back to `COEFFICIENT_TABLE` and count:

```text
16 个词 × 每个词 2 个系数 = 32 个数字
```

*(16 words times 2 coefficients each = 32 numbers.)*

32 numbers — how long did we spend filling them in? And that's only a toy vocabulary of 16
words.

- A vocabulary of 50,000 words is **100,000 numbers**. Who fills those in?
- And even if someone agreed to do it, how would they know 芯片 should be 1.5 rather than 1.4?
- Worse still: **how did we know these numbers were "filled in correctly"?**
  We ran these 10 sentences over and over and stopped when everything was right. Which means
  these numbers were **tuned against the answers** — swap in a different batch of sentences and
  all of them are void.

That "苹果 -> (0.2, 0.9)" in the experiment shows the problem especially well. Why not
(0.5, 0.5)? Why not (0.1, 0.8)? We honestly don't know. We just tried a few and kept whichever
one got everything right.

So the question is very concrete:

> **Who decides these 1.2s, 0.8s and 1.5s?**

We don't want to be the person filling in numbers any more. Next chapter we'll try:
**let the program change these numbers itself.**

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Turn 芯片's coefficient down from 1.5 little by little. At what value does 「苹果芯片很强」
   start getting answered wrong? (Hint: work out that sentence's two scores.)
2. Change **both** coefficients on the 苹果 row to 0 and run it. Does the accuracy change?
   A word that appears in 7 of the sentences and belongs to both categories at once can have
   all-zero coefficients — what does that say about how much we know about this table?
3. Multiply all four coefficients of 新, 很, 这个 and 做成 by 2 and run it. Does the accuracy
   change? Does the gap between 「苹果芯片很强」's two scores change?
4. Suppose the corpus gained one more sentence, 「苹果很香」. Which numbers would you change?
   After changing them, are the original 10 sentences still all correct?
