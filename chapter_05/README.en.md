**English** | [中文](README.md)

# Chapter 5: How big is the error, really?

## The problem in this chapter

The last chapter left us with a question:

> How much, exactly, should we change each time?

To answer it, we first need one thing: **a number that can say "how badly wrong it is".**

Let's look at what we have on hand. `before.py` runs Chapter 4's code unchanged:

```text
  轮次   这把尺子的读数（答错了几句）
    1      1
    2      4
    3      2
    4      0
    5      0
    6      0
    ...
```

*(The columns are "round" and "what this ruler reads (how many sentences were answered wrong)".
From round 4 on, the reading is 0 forever. The program thinks "there's no error left" and stops.)*

After round 4, the reading stays at 0. The program decides there's no error left, so it stops.

The problem is: **this ruler only has 11 marks (0 to 10), and it has hit the top.**

The trained table looks like this (again, output from `before.py`):

```text
  苹果发布新手机     科技   2.0 / 食品  -2.0   赢  4.0
  苹果发布新芯片     科技   3.0 / 食品  -3.0   赢  6.0
  苹果芯片很强       科技   1.0 / 食品  -1.0   赢  2.0
  ...
  这个苹果真甜       科技  -4.0 / 食品   4.0   赢  8.0
```

*(苹果发布新手机: technology 2.0 / food -2.0, winning by 4.0. 苹果发布新芯片: 3.0 / -3.0, winning by
6.0. 苹果芯片很强: 1.0 / -1.0, winning by 2.0. ... 这个苹果真甜: -4.0 / 4.0, winning by 8.0.)*

「苹果芯片很强」 wins by only 2.0 points; 「这个苹果真甜」 wins by 8.0 points.

**One is a barely-scraped win, the other is a comfortable win. But Chapter 4's ruler reads them
exactly the same: both right.**

So there are three things we can't answer:

- which sentence is actually about to lose?
- this table versus another table that is also "0 wrong" — which is better?
- can training still get better?

So what we're looking for is a number like this:

> **It has to be able to say "how badly wrong it is".**
> When the answer is wrong, it has to distinguish "just barely off" from "wildly off";
> when the answer is right, it also has to distinguish "barely right" from "very sure".

## The simplest attempt

The most intuitive move is: **look at how far off it is, and record however far off it is.**

We already have two scores on hand. Subtract them to get "how much this sentence leans
technology":

```text
分数差 = 科技分数 - 食品分数
```

*(score difference = technology score - food score)*

Positive leans technology, negative leans food. (「苹果芯片很强」 is +2.0, 「苹果很甜」 is -2.0.)

Then "the correct answer" is a target value: technology is +1, food is -1. How far the prediction
is from the target is how wrong it is:

```python
def signed_error(words, label, table):
    difference = 科技分数 - 食品分数
    target = 1.0 if label == "科技" else -1.0
    return (difference - target) ** 2
```

Whatever the gap is, square it. This is called **square error**.

It looks perfectly reasonable: the further off, the bigger the number, the worse the mistake.

Experiment 2 in `experiment.py` measures a few sentences with it. Let's look at the result:

```text
  句子                    赢了几分      平方误差       交叉熵   说明
  苹果芯片很强           2.0      1.00     0.127   只赢一点点，最勉强
  苹果发布新手机         4.0      9.00     0.018   赢 4 分，还行
  苹果发布新芯片         6.0     25.00     0.002   赢 6 分，挺稳
  这个苹果真甜           8.0     49.00     0.000   赢 8 分，最有把握
```

*(Columns: sentence, how many points it won by, square error, cross-entropy (交叉熵), and a note.
苹果芯片很强: won by 2.0, square error 1.00, cross-entropy 0.127 — "wins by a tiny bit, the most
reluctant". 苹果发布新手机: 4.0, 9.00, 0.018. 苹果发布新芯片: 6.0, 25.00, 0.002. 这个苹果真甜: 8.0,
49.00, 0.000 — "wins by 8, the most confident".)*

Look at the top and bottom rows:

- 「苹果芯片很强」 is **the most reluctant** (wins by only 2 points), and square error gives it
  **1.00**, calling it the most accurate;
- 「这个苹果真甜」 is **the most confident** (wins by 8 points), and square error gives it
  **49.00**, calling it the worst mistake.

**It says the opposite of the truth.**

The reason is on this line:

```python
target = 1.0
```

Square error wants the "score difference" to be **exactly 1**. So everything past 1 gets counted
as "error" — the more confident, the more wrong.

That makes no sense. The ruler we want says: **the more confident the better**, not "exactly 1.0
is best".

## Experiment

Run `experiment.py`.

**Experiment 1: three rulers, measuring three tables.**

```text
                答错几句        平方误差           交叉熵
  表 A             0       18.60        0.0210
  表 B             1       28.20        0.1359
  表 C             1      738.60        3.8044
```

*(Columns: wrong answers, square error, cross-entropy. Table A: 0, 18.60, 0.0210. Table B: 1,
28.20, 0.1359. Table C: 1, 738.60, 3.8044.)*

(Table A is the table trained in Chapter 4; table B spoils it a little, so that 「苹果芯片很强」
is just barely wrong; table C spoils the same spot badly, so that it's wrong by a mile.)

- Ruler one (how many sentences are wrong): table B and table C both read 1 — **identical**.
- Ruler two (square error): it can tell them apart, but the numbers it gives are hard to explain
  (see above).
- Ruler three (cross-entropy): B and C differ by nearly a factor of 28. The worse the mistake,
  the harder the punishment.

**Experiment 3: how a probability comes out of a score.**

```text
  一个类别的时候：
    分数  -4  ->  概率 0.0180
    分数  -2  ->  概率 0.1192
    分数  -1  ->  概率 0.2689
    分数   0  ->  概率 0.5000
    分数   1  ->  概率 0.7311
    分数   2  ->  概率 0.8808
    分数   4  ->  概率 0.9820

  两个类别的时候：
    分数 [1.7, 1.2]     ->  概率 [0.6225, 0.3775]
    分数 [1.0, -1.0]    ->  概率 [0.8808, 0.1192]
    分数 [-0.1, 0.1]    ->  概率 [0.4502, 0.5498]
    分数 [8.0, 0.0]     ->  概率 [0.9997, 0.0003]
```

*(With one category: score -4 -> probability 0.0180, -2 -> 0.1192, -1 -> 0.2689, 0 -> 0.5000,
1 -> 0.7311, 2 -> 0.8808, 4 -> 0.9820. With two categories: scores [1.7, 1.2] -> probabilities
[0.6225, 0.3775], and so on.)*

**Experiment 4: three tables, two rulers.**

```text
  表                            正确率         交叉熵
  全 0 的表（起点）            5/10      0.6931
  第 4 章的规则训出来的       10/10      0.0210
  本章用损失训出来的          10/10    1.00e-11
```

*(Table, accuracy, cross-entropy: the all-zero table (the starting point) 5/10, 0.6931; the table
trained by Chapter 4's rule 10/10, 0.0210; the table trained by this chapter's loss 10/10,
1.00e-11.)*

Chapter 4's ruler would say: the last two are both 10/10, equally good.
This chapter's ruler says: 0.6931 -> 0.0210 -> 1.00e-11.

**Experiment 5: how much to change each time finally means something.**

```text
  每次改多少           轮数          最终损失     正确率
  0.05           229      8.62e-11    10/10
  0.5             26      1.00e-11    10/10
  2.0              8      2.62e-12    10/10
  5.0              5      4.69e-14    10/10
```

*(Columns: how much we change each time, the number of rounds, the final loss, the accuracy.
0.05 needs 229 rounds and ends at 8.62e-11; 0.5 needs 26 rounds, 1.00e-11; 2.0 needs 8 rounds,
2.62e-12; 5.0 needs 5 rounds, 4.69e-14. All 10/10.)*

We did the same thing back in Chapter 4, and back then all four step sizes produced identical
training trajectories. Not any more: change a little and you need over two hundred rounds; change
a lot and you're there in a few.

## The new mechanism

What this chapter builds comes in three steps.

### Step one: turn scores into probabilities

A score can be any number: -21 is fine, +12.5 is fine. It could even be 1000.
What we need is a number between 0 and 1 that can be read as "how likely is this".

Start with the simplest case: **only one category**. We want to answer "yes" or "no", and we have
just one score in hand.

We want this function to satisfy:

- when the score is 0, the probability should be 0.5 (the two options are even, which is the same
  as not knowing);
- the bigger the score, the closer to 1; the smaller, the closer to 0;
- it never goes outside 0 to 1.

There is a function that looks exactly like this:

```text
概率 = 1 / (1 + e 的 (-分数) 次方)
```

*(probability = 1 / (1 + e to the power of minus the score))*

(`e` is about 2.718, and "e to the x" is written `math.exp(x)` in Python.)

Plug a few numbers in and you get the first table of experiment 3: score 0 gives 0.5, score 1
gives 0.7311, score -2 gives 0.1192, score 4 gives 0.9820. This function is called **sigmoid**.

**With two categories (and later, more than two)**, we want the two probabilities to "compete" and
to add up to exactly 1. The natural move: turn every score into `e to the score`, then divide by
their total.

```text
概率_1 = e 的 分数1 次方 / (e 的 分数1 次方 + e 的 分数2 次方)
概率_2 = e 的 分数2 次方 / (e 的 分数1 次方 + e 的 分数2 次方)
```

*(probability_1 = e to score1 / (e to score1 + e to score2), and likewise for the second.)*

Put the scores [1.0, -1.0] in and you get `e^1 / (e^1 + e^-1)` and `e^-1 / (e^1 + e^-1)`, which is
0.8808 and 0.1192.

This method is called **softmax**.

It's really the same thing as sigmoid: when the scores are `[d, 0]`, the first probability softmax
computes is exactly sigmoid(d) — verified in experiment 3:

```text
    d = -2.0:  softmax 0.119203   那个式子 0.119203
    d =  0.0:  softmax 0.500000   那个式子 0.500000
    d =  1.0:  softmax 0.731059   那个式子 0.731059
```

*(At d = -2.0, softmax gives 0.119203 and "that formula" — the sigmoid formula above — gives
0.119203; at d = 0.0 both give 0.500000; at d = 1.0 both give 0.731059.)*

### Step two: from probability to "how big is the error"

Now every sentence has a probability: **the probability of the correct answer.**

- it equals 1: very certain, not wrong at all;
- it equals 0.5: purely guessing;
- it approaches 0: wrong by a mile.

We want to turn this probability into an "error size". The most direct idea is `1 - probability`,
but then "wrong by a mile" costs far too little: a probability of 0.01 and a probability of 0.0001
differ by only 0.0099, and those two cases are obviously orders of magnitude apart.

So we use this:

```text
损失 = -log(正确答案的概率)
```

*(loss = -log(probability of the correct answer))*

`log` is the natural logarithm (written `math.log` in Python). Plug a few probabilities in and
look:

```text
  概率 1.0     ->  损失 0
  概率 0.99    ->  损失 0.0101
  概率 0.9     ->  损失 0.1054
  概率 0.5     ->  损失 0.6931
  概率 0.1     ->  损失 2.3026
  概率 0.01    ->  损失 4.6052
  概率 0.001   ->  损失 6.9078
```

*(probability 1.0 -> loss 0; 0.99 -> 0.0101; 0.9 -> 0.1054; 0.5 -> 0.6931; 0.1 -> 2.3026;
0.01 -> 4.6052; 0.001 -> 6.9078.)*

The three properties are exactly what we wanted:

1. **Right but not sure enough, and the loss isn't 0.** A probability of 0.9 still carries 0.1054
   of loss — there's room to improve, and training still knows which way to go.
2. **The worse the mistake, the faster it climbs.** Probability drops from 0.5 to 0.1 and loss
   climbs from 0.69 to 2.30; drop to 0.01 and loss is 4.61. It doesn't grow linearly — the closer
   you get to "completely wrong", the heavier the penalty.
   (We'll use this property again in Chapter 6: when we work out which direction each parameter
   should move in, the answer that comes out will depend on "how badly wrong this sentence is".)
3. **As the probability approaches 1, the loss approaches 0**, and it never turns negative.

Compute this loss for every sentence in the corpus and take the average, and you get **a single
number**. That number is the ruler we've been looking for.

Its name is **loss**, and this way of computing it is called **cross-entropy**.

### Step three: why it beats "counting wrong sentences"

Because it's continuous, and it has no ceiling.

"Out of 10 sentences, how many were right" has only 11 possible values. When two tables are both
10/10, it has nothing left to say. Loss is different: it can keep going down, 0.6931 -> 0.0210 ->
0.00000000001, and it can see the difference at every step.

That's why it can answer "which table is better" and accuracy can't.

## Python implementation

The full code is in `after.py`. The core is two functions.

```python
def softmax(scores):
    biggest = max(scores)
    exponentials = [math.exp(value - biggest) for value in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]
```

That `biggest` line is a guard: if a score is 1000, `math.exp(1000)` blows up outright (a float
can't hold it). Subtract the maximum from every score first and all the `e` exponents are at most
0, and the result doesn't change — because numerator and denominator get multiplied by the same
number.

```python
def loss_of(words, label):
    correct_index = CATEGORIES.index(label)
    return -math.log(probability_of(words, correct_index))
```

A sentence's loss is just the negative logarithm of "the probability of the correct answer".

And then we used it to do something Chapter 4 couldn't:

```python
def try_adjust_one_parameter(corpus, word, category_index, amount, current_loss):
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
    loss = average_loss(corpus)
    for word in VOCABULARY:
        for category_index in (0, 1):
            loss = try_adjust_one_parameter(corpus, word, category_index, amount, loss)
    return loss, changed_count
```

These two functions are not clever at all. They are pure **trial**:

- go through every parameter, try adding a bit, recompute the whole corpus, and see whether the
  loss went up or down;
- then try subtracting a bit, and compute it all again the same way;
- whichever side is lower, use it; if both are higher, back off and leave it as it was.

It's dumb, but **it only needs one thing to work: a ruler that can tell "better" from "worse".**

Chapter 4 didn't have that ruler, so it couldn't do this.

Running it looks like this:

```text
  轮次   损失        这一轮改了几个参数   正确率
    1    0.1329                    30   10/10
    2    0.0443                    30   10/10
    3    0.0144                    29   10/10
    4    0.0053                    28   10/10
    ...
   25    1.00e-11                   2   10/10
   26    1.00e-11                   0   10/10
```

*(Columns: round, loss, how many parameters changed this round, accuracy. Round 1: loss 0.1329,
30 parameters changed, 10/10. Round 2: 0.0443, 30, 10/10. Round 3: 0.0144, 29, 10/10. Round 4:
0.0053, 28, 10/10. ... Round 25: 1.00e-11, 2, 10/10. Round 26: 1.00e-11, 0, 10/10.)*

## What it solves

**First, it builds a continuous ruler for "error".**

| | Chapter 4's ruler | this chapter's ruler |
|---|---|---|
| what it looks like | integers 0 to 10 | any positive number |
| the starting point (all-zero table) | 5/10 | 0.6931 |
| the table trained in Chapter 4 | 10/10 | 0.0210 |
| the table trained in this chapter | 10/10 | 1.00e-11 |
| can it tell the last two apart | no | yes |

**Second, "which table is better" has an answer for the first time.**

At the end of Chapter 4 we had three tables that all got 10 sentences right, and nobody could say
which was better. Now measure them side by side: 0.0210 and 1.00e-11. The table trained in this
chapter has pushed every sentence to a nearly certain position.

**Third, the question Chapter 4 left behind has an answer.**

How much to change each time? Experiment 5: change by 0.05 and it takes 229 rounds; change by 5.0
and it takes 5.

What's interesting is that the answer itself doesn't matter — what matters is that **the question
can now be asked at all**. Chapter 4 couldn't ask it, because back then we couldn't even measure
whether a change was good or bad.

## What it still can't solve

Go back and look at that `adjust_weights_once` function. This is how it works:

```text
对每一个参数：
    加一点，把 10 句话全算一遍，看损失
    减一点，把 10 句话全算一遍，看损失
    选好的那个
```

*(For every parameter: add a bit, compute all 10 sentences, look at the loss; subtract a bit,
compute all 10 sentences, look at the loss; keep whichever is better.)*

Count it up: our table has 32 parameters, each parameter gets tried 2 times, and every trial
recomputes the entire corpus. **One round is 68 full computations.** The training run above went
26 rounds, more than 1,700 computations in total.

32 parameters is fine. A thousand-odd computations is a blink.

But what happens when the number of parameters grows?

- our table: 32 parameters -> 68 computations per round.
- a real model (the kind you've used): **hundreds of millions to hundreds of billions of
  parameters**.
- Do it the same way and one round takes tens of billions of computations. And that's one round.

Worse, this method is **very wasteful**: while it tries parameter 1, the other 31 parameters sit
still and wait. They're obviously related to each other (苹果 and 芯片 have to pull together to
push 「苹果芯片很强」 over the line, for instance), but our method always moves exactly one at a
time.

So the question is very concrete:

> **Once there are many parameters, trying them one at a time is far too slow.**

We need a way to **compute, in one shot, which direction each parameter should move in and by how
much**, instead of trying them one by one.

Next chapter we'll build that thing.

## Exercises

See `exercises.en.md`. The 5 most important ones:

1. Work it out with pen and paper: under table A, what is the probability of the correct answer
   for 「苹果芯片很强」? What is the loss? (The scores are technology 1.0 / food -1.0. When you're
   done, run `experiment.py` and check.)
2. Change `ADJUST_AMOUNT` from 0.5 to 0.05 and to 5.0, running `after.py` each time. How much do
   the round counts differ? How much does the final loss differ?
3. Replace the `-math.log(...)` in `loss_of` with `1 - probability` (an "error rate"). Can training
   still push the accuracy to 10/10? Does the loss curve look the same?
4. Why does `softmax` subtract the maximum first? Try deleting that line, then call
   `softmax([1000.0, 0.0])` and see what happens.
5. Right now `adjust_weights_once` only accepts a change when the loss really goes down. What does
   training look like if you change it to "accept as long as the loss didn't go up" (`<=`)?
