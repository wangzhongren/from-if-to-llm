**English** | [中文](README.md)

# Chapter 4: Letting the program change the weights itself

## The problem in this chapter

The last chapter left us with a question:

> Who decides these 1.2s, 0.8s and 1.5s?

The table in Chapter 3 has 32 numbers in it, all filled in by hand. And the method was not
dignified: run it, see which sentence is wrong, change one number, run it again, keep going
until it's 10/10.

`before.py` puts that on display for you:

```text
  表 A（人填的）    正确 10/10
  表 B（随便填的）  正确 4/10

表 A 是怎么来的？我们跑一遍、看哪句错了、改一个数字、再跑一遍……
改到 10/10 就停手。也就是说：这 32 个数字是我们'看着答案'调出来的。

程序在这件事里干了什么？它只负责算分数和比大小。
哪个数字该改成多少，它一点想法都没有。
```

*(Table A, the human-filled one, scores 10/10; table B, filled in at random, scores 4/10. The
text underneath: how did table A come about? We ran it, saw which sentence was wrong, changed
one number, ran it again... and stopped at 10/10. Which means those 32 numbers were tuned while
looking at the answers. What did the program do in all of this? It only computed the scores and
compared them. How much any number should change to, it has no idea.)*

Table B was filled in at random, and its shape is identical to table A: 16 rows, two numbers per
row. The program can't tell the two tables apart — **it doesn't even know table B is worse than
table A.**

There's one thing we want: **let the program change these numbers itself.**

## The simplest attempt

First ask: when we change a number, what are we going on?

We see 「苹果很甜」 answered as "technology" and we think: "甜 is obviously evidence for the food
side, turn it up a bit; 苹果 is meddling too much on the technology side, turn it down a bit."

That line of thinking needs no mathematics at all. It has exactly one prerequisite: **we know
what the right answer is.** When it's wrong, nudge a little in the direction of the right answer.

So the rule can be written very plainly:

> If the sentence is answered wrong, turn the numbers for the correct category up a bit, and
> turn the numbers for the category we answered down a bit.

And when it's right? **Do nothing.** It's already right; there's no reason to touch it.

For example, if 「苹果很甜」 is answered "technology" and the true answer is "food":

```text
苹果：科技 -1    食品 +1
很：  科技 -1    食品 +1
甜：  科技 -1    食品 +1
```

*(苹果, 很 and 甜 each move one step toward food and one step back from technology. Words that
don't appear in this sentence — 芯片, 香蕉 — aren't touched at all.)*

Every word that appears in the sentence takes one step toward "food" and one step back from
"technology". Words that don't appear in this sentence (芯片, 香蕉) don't move at all.

## Experiment

Run `after.py`. The starting point is **an all-zero table** — we fill in nothing:

```text
==================================================================
起点：所有词的权重都是 0
==================================================================
  正确：5/10
  解释：全 0 的时候两个分数都是 0，按规矩先算科技——
        5 句科技正好被蒙对了，5 句食品全错。

==================================================================
训练：错了就改，对了不动
==================================================================
  轮次   答错    正确
    1     1      7/10  #
    2     4      9/10  ####
    3     2     10/10  ##
    4     0     10/10
```

*(The starting point: with every weight 0, both scores are 0, and by the rule technology is
counted first — so the 5 technology sentences are right by accident and all 5 food sentences are
wrong: 5/10.*

*Then training: "wrong means change, right means leave alone". Round 1 has 1 wrong answer, 7/10;
round 2 has 4 wrong, 9/10; round 3 has 2 wrong, 10/10; round 4 has 0 wrong, 10/10. The bars to
the right are just a picture of the wrong counts. The comment lines underneath say "wrong count"
and "correct".)*

All correct after 4 rounds. The trained table looks like this:

```text
  苹果     科技  -1.0   食品   1.0
  发布     科技   1.0   食品  -1.0
  新       科技   1.0   食品  -1.0
  手机     科技   1.0   食品  -1.0
  芯片     科技   2.0   食品  -2.0
  电脑     科技   0.0   食品   0.0
  华为     科技   0.0   食品   0.0
  小米     科技   0.0   食品   0.0
  好吃     科技  -2.0   食品   2.0
  很       科技   0.0   食品   0.0
  甜       科技  -1.0   食品   1.0
  香蕉     科技   0.0   食品   0.0
  这个     科技  -1.0   食品   1.0
  真       科技  -1.0   食品   1.0
  做成     科技  -1.0   食品   1.0
  派       科技  -1.0   食品   1.0
```

*(Each row: a word, its technology weight, its food weight. Half the words ended at 0.0 on both
sides; 芯片 reached +2.0 / -2.0; 好吃 -2.0 / +2.0.)*

A few things are worth stopping for:

- **Round 2 has more wrong sentences than round 1** (1 -> 4). That's because it changes and
  judges within the same round: the weights changed in the first half of the round affect the
  judgements in the second half. It's not surprising, just something to get used to.
- **Half the words are 0.** 电脑, 华为, 小米, 香蕉 and 很 were all adjusted, and they all came back
  to 0. The reason is simple: the program only acts when the answer is **wrong**. By the time it
  reaches those sentences, the weights are already good enough and those sentences are never
  wrong — **never wrong, never adjusted.**
- 芯片 is 2.0 and 很 is 0.0. In the hand-filled table, 芯片 was 1.5 and 很 was 0.3. The two tables
  look nothing alike, and both get all 10 sentences right.

Now the thing in `experiment.py` that we didn't expect. At the start of Chapter 4, we casually
set "how much to change each time" to 1.0. So let's try other numbers:

```text
  每次改多少    每轮答错的句数                                        最终正确率
  0.01           1  4  2  0  0  0  0  0  0  0      10/10
  0.1            1  4  2  0  0  0  0  0  0  0      10/10
  1.0            1  4  2  0  0  0  0  0  0  0      10/10
  10.0           1  4  2  0  0  0  0  0  0  0      10/10
```

*(Columns: how much we change each time, the number of wrong sentences per round, and the final
accuracy. All four rows read exactly the same: 1 4 2 0 0 0 0 0 0 0, and 10/10.*

*The note underneath: four identical rows. 0.01 and 10.0 differ by a factor of 1000, and the
training trajectory isn't off by a single character.*

*The reason isn't hard: we decide the category by **which of the two scores is larger**.
Multiply every weight by 100 at once and both scores get multiplied by 100 at once; which is
larger doesn't change at all.*

*So under Chapter 4's rule, "how much to change each time" **doesn't affect right or wrong**. It
affects exactly one thing: how big the weights are.)*

Four identical rows. 0.01 and 10.0 differ by a factor of 1000, and the training trajectory
doesn't differ by a single character.

The reason isn't hard to see: we decide the category by **which of the two scores is larger**.
Multiply every weight by 100 and both scores get multiplied by 100 as well — which one is larger
doesn't change at all.

So under Chapter 4's rule, "how much to change each time" **doesn't affect right or wrong.** It
affects exactly one thing: the size of the weights.

## The new mechanism

The things this chapter introduces are already written above. Now let's give them two names.

**First name: weight.**

Those numbers in Chapter 3 we have been calling "coefficients". From this chapter on, we call
them **weights**. The name changes because their status changed:

- in Chapter 3 they were "knowledge we typed in";
- in Chapter 4 they are "something the program maintains itself".

**Second name: training.**

The action "when the answer is wrong, go change the weights by the rule" — we call that
**training**. Walking the whole corpus once is called a **round**. We'll keep using these words
until the end of the book.

The rule itself, written out in full, is this:

```python
def adjust(words, correct_category, wrong_category, step):
    for word in words:
        WEIGHT_TABLE[word][correct_index] += step
        WEIGHT_TABLE[word][wrong_index] -= step
```

Why is it reasonable? You can think of it like this:

**Every word is two pieces of evidence.** The word 甜 appearing is one piece of evidence that
"this sentence is food", and at the same time a piece of counter-evidence that "this sentence is
technology". Getting it wrong means we overestimated one piece of evidence and underestimated the
other, so push the overestimated one down a little and the underestimated one up a little.

What's worth noticing is that it **needs no mathematics**. We don't know "how badly wrong" it
was, and we don't know in which direction or by how much to adjust — we only know **what the
answer is**, and **which sentence was wrong**. With just those two pieces of information, 32
numbers grew themselves.

## Python implementation

The full code is in `after.py`. There are three core pieces.

```python
WEIGHT_TABLE = {word: [0.0, 0.0] for word in VOCABULARY}
```

An all-zero table. The order of the two numbers is fixed: [technology, food]. So
`WEIGHT_TABLE["芯片"] = [2.0, -2.0]` means "芯片 gives technology 2.0 and food -2.0".

```python
def predict(words):
    scores = score(words)
    return CATEGORIES[0] if scores[0] >= scores[1] else CATEGORIES[1]
```

The decision rule hasn't changed: the higher score wins. When the two scores are equal,
technology counts first — this little convention for "what to do about a tie" will come up every
day when training has just started (in an all-zero table every sentence is 0 : 0), so it's worth
writing down clearly.

```python
def train_one_round(corpus, step):
    wrong_count = 0
    for sentence, label in corpus:
        words = split_sentence(sentence)
        answer = predict(words)
        if answer == label:
            continue
        wrong_count += 1
        adjust(words, correct_category=label, wrong_category=answer, step=step)
    return wrong_count
```

One round is "walk the corpus from beginning to end". While it walks, the weights keep changing —
so the first half of a round and the second half of the same round aren't really using the same
table.

## What it solves

The Chapter 3 table had 32 numbers and we filled them in. This chapter's table, **we didn't fill
in a single one**:

```text
  起点         5/10
  4 轮之后     10/10
```

*(Starting point 5/10; after 4 rounds, 10/10.)*

And the answer it found is **not the same** as the one we filled in:

| word | what we filled in | what the program trained |
|---|---|---|
| 苹果 | technology 0.2 / food 0.9 | technology -1.0 / food 1.0 |
| 芯片 | technology 1.5 / food 0.0 | technology 2.0 / food -2.0 |
| 很 | technology 0.0 / food 0.3 | technology 0.0 / food 0.0 |

The two tables' numbers are completely different, and the score is the same. **Which shows that
"a table that gets everything right" is not unique** — remember this, we'll need it next chapter.

The other change is the program's identity. The Chapter 3 program worked like this: we change the
numbers, it computes the scores. This chapter's program: **it changes the numbers itself.**

We supplied only three things:

1. a batch of sentences with the answers marked on them;
2. one rule: "wrong means change";
3. one number: how much to change each time.

## What it still can't solve

Of the three things, the third is the most suspicious: **how much to change each time?**

We set it to 1.0 on a whim. Experiment 1 showed that it apparently **doesn't affect right or
wrong** — 0.01 and 10.0 produced identical training trajectories. So what does it affect?

```text
  每次改多少        正确率      权重最大的绝对值
  0.01         10/10    0.02
  0.1          10/10    0.20
  1.0          10/10    2.00
  10.0         10/10    20.00
```

*(Columns: how much we change each time, the accuracy, and the largest absolute value among the
weights. 0.01 -> 0.02; 0.1 -> 0.20; 1.0 -> 2.00; 10.0 -> 20.00. Every row is 10/10.)*

It affects the size of the weights. But what does "the size of the weights" mean? We can't answer
that.

What's worse is this:

```text
  第 3 章手填的表                科技    1.7 / 食品    1.2
  训练出来的表（每次改 1.0）     科技    1.0 / 食品   -1.0
  训练出来的表（每次改 10.0）    科技   10.0 / 食品  -10.0
```

*(The hand-filled Chapter 3 table: technology 1.7 / food 1.2. The trained table with 1.0 per
change: technology 1.0 / food -1.0. The trained table with 10.0 per change: technology 10.0 /
food -10.0.)*

All three tables are 10/10. If you ask "which one is better", we have **nothing at all** on hand
that can answer you.

Our only ruler is "how many of the 10 sentences were right". It has just 11 possible values, from
0 to 10, and all three tables are at 10. The ruler has run out.

So the question is very concrete:

> **How much, exactly, should we change each time?**

So the question is very concrete:

> **How much, exactly, should we change each time?**

To answer it, we first have to be able to say "how badly wrong this sentence is" — not "it's
wrong", but "how wrong". Right now our entire understanding of the error is "苹果芯片很强 was
answered as food".

Next chapter we build a ruler for "error".

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Change `STEP` to 0.001 and to 1000.0 and run `after.py` each time. Does the number of training
   rounds change? Why? (Guess first, then run.)
2. After training, change the three words 电脑, 华为 and 小米 by hand to `[5.0, -5.0]` and run
   `measure` again. Does the accuracy change? Why?
3. Delete the `wrong_index` line from `adjust` (only add points to the correct side, never
   subtract from the wrong side). Can it still train to 10/10?
4. Deliberately change the label of 「苹果很甜」 to "technology" and train again. How many rounds
   does the program need to get all 10 sentences right? Can it really do it?
