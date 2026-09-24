**English** | [中文](README.md)

# Chapter 12: A word can't have only one meaning

## The problem in this chapter

The last chapter ended on a question: **a word can only ever have one meaning.**

Specifically this. After Chapter 11 trained that table, we looked at 苹果's neighbours:

```text
  苹果   → 香蕉 0.82  华为 0.61  好吃 0.61  小米 0.61  甜 0.56
```

*(苹果 "apple"'s five nearest words after Chapter 11's training — three from the food side (香蕉 "banana" 0.82, 好吃 "delicious" 0.61, 甜 "sweet" 0.56) and two from the tech side (华为 "Huawei" 0.61, 小米 "Xiaomi" 0.61).)*

香蕉, 好吃 and 甜 are on the food side; 华为 and 小米 are on the tech side. It gets pulled from both sides at once.

That looks perfectly reasonable. But our corpus contains both kinds of sentence:

```text
苹果 很 好吃          这里说的是水果
苹果 发布 新 手机      这里说的是公司
```

*(苹果 很 好吃 — "apples are delicious", here it is the fruit; 苹果 发布 新 手机 — "Apple shipped a new phone", here it is the company.)*

Chapter 11's rule is "one word, one vector". So the 苹果 in those two sentences gets the same group of numbers — not "roughly the same", the same thing.

`before.py` lays this out on the table:

```text
    句子 A：苹果 很 好吃          （水果）
    句子 B：苹果 发布 新 手机     （公司）

  A 句里 苹果 的向量：
    [+0.73  +0.71  +0.04  -0.36  -1.07  +0.23  -0.73  +0.67  -0.10  +0.05  -0.12  +0.17  +1.02  -0.16  -0.75  -0.04]
  B 句里 苹果 的向量：
    [+0.73  +0.71  +0.04  -0.36  -1.07  +0.23  -0.73  +0.67  -0.10  +0.05  -0.12  +0.17  +1.02  -0.16  -0.75  -0.04]

  两个向量的最大差值 = 0.0000000
  余弦相似度 = 1.0000
```

*(The printout compares sentence A 苹果 很 好吃 (the fruit) with sentence B 苹果 发布 新 手机 (the company). Both print an identical 16-number vector; the largest difference between the two vectors is 0.0000000 and their cosine similarity is 1.0000.)*

The consequences are concrete. Ask "which word is most like 苹果":

```text
  在 A 句里问：香蕉 +0.73  华为 +0.58  甜 +0.57  小米 +0.48  好吃 +0.48  派 +0.48
  在 B 句里问：香蕉 +0.73  华为 +0.58  甜 +0.57  小米 +0.48  好吃 +0.48  派 +0.48
```

*(The same question asked inside sentence A and inside sentence B returns an identical list: 香蕉 "banana" +0.73, 华为 "Huawei" +0.58, 甜 "sweet" +0.57, 小米 "Xiaomi" +0.48, 好吃 "delicious" +0.48, 派 "pie" +0.48.)*

The two lines are identical.

More concretely still: suppose something downstream has to decide whether "the 苹果 here" is a fruit or a company.
The information the model can use is 苹果's vector, and its similarity to other words:

```text
                     好吃    甜      香蕉    发布    手机    芯片
  A 句里的 苹果      +0.48   +0.57   +0.73   -0.05   -0.11   -0.18
  B 句里的 苹果      +0.48   +0.57   +0.73   -0.05   -0.11   -0.18
```

*(The same comparison for both sentences, against the columns 好吃 "delicious", 甜 "sweet", 香蕉 "banana", 发布 "release", 手机 "phone", 芯片 "chip". The 苹果 in A and the 苹果 in B have exactly the same similarity to every one of them.)*

Still two identical lines.

**The trouble isn't that we didn't train long enough, and it isn't that the dimension isn't high enough.
It's that we nailed a word down to a single position — and one position can only hold one meaning.**

If a word means different things in two sentences, then its **representation** in those two sentences ought to differ.
What this chapter builds is exactly that "representation that changes with the sentence".

## The simplest attempt

So how do we compute a representation for "苹果" that changes with the sentence?

The dumbest way, and the first one anyone would think of: **don't use 苹果's own vector, use the vectors of the words around 苹果.**

Look:

```text
苹果 很 好吃          →  用 (很 + 好吃) / 2
苹果 发布 新 手机      →  用 (发布 + 新 + 手机) / 3
```

*(In 苹果 很 好吃, use the average of 很 "very" and 好吃 "delicious"; in 苹果 发布 新 手机, use the average of 发布 "release", 新 "new" and 手机 "phone".)*

In the first sentence what stands beside 苹果 is 很 and 好吃 — both food-side words, so the average leans food-wards on its own.
In the second sentence what stands beside it is 发布, 新 and 手机 — all tech-side words, so the average leans tech-wards.

We don't need to teach the model "苹果 has two meanings" at all. **Just stop looking at 苹果 itself, and look at who is standing next to it.**

The words around the place where a word sits — let's give them a name. From now on we call them that word's **context**.

The vector you get by averaging the vectors of every word in the context, we call the **context vector** of that position.

Do note how dumb this scheme is: it doesn't use 苹果's own vector at all.
Even if you replaced 苹果 with a word you had never seen, as long as what surrounds it is 很 and 好吃, the context vector still leans food-wards.

One more detail to be clear about: **averaging** is the only operation used here.
It is so simple that it is a little suspicious — and we'll see exactly where it's suspicious in a moment.

## Experiment

`after.py` first trains Chapter 11's table just as it was (the corpus is extended to 25 sentences and 33 words, because we need to add the long sentences and the new words they bring with them). Then it does one thing:
it computes two different representations for the same 苹果.

```text
  A 句里 苹果 的上下文向量 = (很 + 好吃) / 2
    [+0.36  -0.67  +0.24  +0.04  +0.31  +0.51  -0.31  -0.14  +0.03  +0.22  -0.21  +0.13  +0.62  -0.37  -0.54  -0.40]
  B 句里 苹果 的上下文向量 = (发布 + 新 + 手机) / 3
    [-0.68  -0.47  -0.79  -0.07  -0.12  +0.16  -0.17  +0.59  -0.27  -0.38  -0.02  -0.08  +0.21  -0.02  +0.58  +0.45]

  两个向量的余弦相似度 = -0.2387
  而两句话里的 苹果 自己的那个向量，余弦相似度 = 1.0000

  A 句里 苹果 最像       → 很 +0.74  好吃 +0.72  甜 +0.63  真 +0.59  这个 +0.45  香蕉 +0.35
  B 句里 苹果 最像       → 电脑 +0.65  新 +0.63  发布 +0.62  的 +0.61  芯片 +0.59  手机 +0.48
```

*(A's context vector for 苹果 is the average of 很 and 好吃; B's is the average of 发布, 新 and 手机. The two context vectors have a cosine similarity of −0.2387 — while 苹果's own vector in the two sentences has a cosine similarity of 1.0000. The list below each one is the words it is most similar to: in A, 很 "very" +0.74 and 好吃 "delicious" +0.72; in B, 电脑 "computer" +0.65, 新 "new" +0.63, 发布 "release" +0.62 and 的 (the possessive particle) +0.61.)*

From 1.0000 to −0.2387. **One word, one ID, one model — and the representations at two positions are now two different things.**

The words it is most similar to changed along with it: in sentence A they are 很, 好吃, 甜; in sentence B they are 电脑, 新, 发布, 芯片.

`experiment.py` measures this difference on a real little task.

The task is this: given a position, decide whether it means the fruit or the company.
The method is to compute its average similarity to one set of food words and one set of tech words, and take whichever is higher.

First, with Chapter 11's static representation (that is, "one word, one vector"):

```text
  句子                                       食品分  科技分  判定     正确答案
  --------------------------------------------------------------------
  苹果 很 好吃                             +0.25  +0.06  水果  ✓      水果
  苹果 很 甜                               +0.25  +0.06  水果  ✓      水果
  苹果 发布 新 手机                        +0.25  +0.06  水果  ✗      公司
  苹果 发布 新 芯片                        +0.25  +0.06  水果  ✗      公司
  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 +0.25  +0.06  水果  ✗      公司

  答对 2 / 5。
  注意「食品分」那一列：五个句子全都是 +0.25，一个字都没变。
  因为静态表示里，这五个「苹果」本来就是同一个向量。
  模型手上只有这一个答案，它只能五句话都答同一个。
```

*(The five sentences are 苹果 很 好吃, 苹果 很 甜, 苹果 发布 新 手机, 苹果 发布 新 芯片, and the long sentence 我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 — "I saw the new phone Apple just released at the mall yesterday". The columns are the food score, the tech score, the verdict and the correct answer. The food score is +0.25 on all five rows, so all five get the same verdict, and 2 of 5 come out right.)*

2 out of 5. **And note that "food score" column — all five sentences give +0.25, not moving even in the first decimal place.**
The model holds only that one number, so it can only answer the same thing to all five sentences.

Now switch to the context vector and change nothing else:

```text
  句子                                       食品分  科技分  判定     正确答案
  --------------------------------------------------------------------
  苹果 很 好吃                             +0.44  -0.20  水果  ✓      水果
  苹果 很 甜                               +0.45  -0.21  水果  ✓      水果
  苹果 发布 新 手机                        -0.23  +0.50  公司  ✓      公司
  苹果 发布 新 芯片                        -0.17  +0.47  公司  ✓      公司
  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 -0.39  +0.31  公司  ✓      公司

  答对 5 / 5。
```

*(Same five sentences, same task. The food score is now +0.44 and +0.45 for the two fruit sentences and −0.23, −0.17, −0.39 for the three company sentences, so the verdicts flip to match, and it gets 5 of 5.)*

Put the two representations side by side, looking at that "food score minus tech score" gap:

```text
  句子                                     静态表示   上下文表示
  --------------------------------------------------------------
  苹果 很 好吃                             +0.188      +0.634
  苹果 很 甜                               +0.188      +0.666
  苹果 发布 新 手机                        +0.188      -0.729
  苹果 发布 新 芯片                        +0.188      -0.639
  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 +0.188      -0.696

  静态表示那一列是个常数；上下文表示那一列跟着句子正负翻面。
```

*(The static column is +0.188 on every row. The context column is +0.634 and +0.666 on the two fruit sentences and −0.729, −0.639, −0.696 on the three company sentences — it flips sign with the sentence.)*

The same table, the same task, and only 苹果's representation swapped: 2/5 becomes 5/5.

## The new mechanism

There is one new thing in this chapter:

> **The representation at a position = the average of the vectors of all the words in its context.**

As a formula:

```text
context(i) = (1 / n) * Σ v_j        （j 取遍句子里除第 i 个词之外的所有词）
```

*(context(i) = (1 / n) × Σ v_j, where j runs over every word in the sentence except the i-th one.)*

In Python:

```python
def context_vector(table_data, sentence, position):
    ids = [VOCAB.index(word) for word in sentence.split()]
    others = [ids[i] for i in range(len(ids)) if i != position]
    return table_data[others].mean(axis=0)
```

Three lines. That is the whole of what is new in this chapter.

It is **a separate thing** from Chapter 11's representation — don't mix the two up:

| | Chapter 11's representation | Chapter 12's representation |
|---|---|---|
| How many representations per word | One | One per position |
| Where the representation comes from | Table lookup | Table lookup + averaging the context |
| The 苹果 in 「苹果 很 好吃」 | `表[苹果]` | `(表[很] + 表[好吃]) / 2` |
| The 苹果 in 「苹果 发布 新 手机」 | `表[苹果]` | `(表[发布] + 表[新] + 表[手机]) / 3` |
| Are the 苹果 in the two sentences the same | Yes | No |

Note the bottom half of that table: **the 表 in those expressions is still the same table, not one character of it changed.**
We only changed how we use it — from "look up one word" to "look up the words around it and average them".

One more thing: the longer the sentence, the more of the context vector comes from words that have nothing to do with the current word.
That turns into a problem very shortly.

## Python implementation

The full code is in `after.py`.

**Step one is still training that table**, exactly as in Chapter 11 — because the context vector is looked up out of the table, and if the table is bad, what you average out of it will be bad too.

The corpus is extended to 25 sentences and 33 words: Chapter 11's 13 short sentences, the two long sentences that Chapters 13–16 will need, plus a few supporting sentences (so that the new words in the long sentences have enough places to appear in). The vocabulary is taken straight out of the corpus:

```python
VOCAB = sorted({word for sentence in SENTENCES for word in sentence.split()})
```

```text
  句子 25 条，词 33 个，维度 16
  训练样本 200 条，训练 600 轮之后的 loss = 1.4998
```

*(The corpus is 25 sentences, 33 words, dimension 16. There are 200 training samples, and after 600 epochs the loss is 1.4998.)*

**Step two is the three lines** of `context_vector` above.

**Step three: use it on different positions of the same word.** The crucial word here is "position":

```python
vector_a = context_vector(table.data, FOOD_SENTENCE, 0)   # 苹果 很 好吃
vector_b = context_vector(table.data, TECH_SENTENCE, 0)   # 苹果 发布 新 手机
```

Same word 苹果, same vector table; only because the sentences the positions sit in differ do the two calls return two different vectors.

## What it solves

| | Chapter 11's representation | Chapter 12's representation |
|---|---|---|
| Cosine similarity of the two 苹果 | 1.0000 (the same) | −0.2387 |
| Accuracy on that little task | 2 / 5 | 5 / 5 |
| "The word most like 苹果" | One answer for both sentences | One answer per sentence |
| Does the representation depend on the sentence it is in | No | Yes |

What this chapter really gets is that last row: **for the first time, a representation is tied to "the sentence it sits in".**

That sounds unremarkable, but it unties a very hard knot: before this, however many meanings a word had, the model could only hold one.
Now every **position** has a representation of its own — a word gets as many representations as there are sentences it appears in.

And the scheme is absurdly dumb: three lines of code, no new parameters, no new training. All it does is "look the table up in a different place". Which brings a bonus — **it can even handle a word it has never seen**: as long as 很 and 好吃 are standing around 苹果, the representation at that position will lean food-wards, even if the table doesn't contain the word 苹果 at all.

But "dumb" has a price. The price is below.

## What it still can't solve

Back to this chapter's algorithm itself: **it averages the words in the context — that is, it treats them all alike.**

`after.py` measures this with this chapter's long sentence:

```text
  我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机

  句中除 苹果 之外还有 10 个词：我 昨天 在 商场 看到 刚刚 发布 的 新 手机
  它们的权重是这样的：
    我 0.10  昨天 0.10  在 0.10  商场 0.10  看到 0.10  刚刚 0.10  发布 0.10  的 0.10  新 0.10  手机 0.10

  每个词的权重都一样。「昨天」和「发布」各占一份，一模一样。
```

*(The long sentence 我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 — "I saw the new phone Apple just released at the mall yesterday". Apart from 苹果 it holds 10 words: 我 "I", 昨天 "yesterday", 在 "at", 商场 "mall", 看到 "saw", 刚刚 "just", 发布 "release", 的 (the possessive/attributive particle, roughly "'s" or "of"), 新 "new", 手机 "phone". The block prints the weight each of them carries — and every single one carries 0.10.)*

Look at that weight table. **「我」 is 0.10, 「的」 is 0.10, 「发布」 is 0.10 — all the same.**

The averaging scheme's plan is: let every word put in 1/10 of the effort, and together they come out roughly right.
But there is a problem here that it cannot solve on its own:

```text
  平均出来的向量，最像哪几个词？
    的 +0.48  电脑 +0.38  新 +0.38  芯片 +0.37  手机 +0.31  小米 +0.29  看到 +0.27  发布 +0.27

  排第一的是「的」。
```

*(Which words is the averaged vector most similar to? 的 (the possessive particle) +0.48, 电脑 "computer" +0.38, 新 "new" +0.38, 芯片 "chip" +0.37, 手机 "phone" +0.31, 小米 "Xiaomi" +0.29, 看到 "saw" +0.27, 发布 "release" +0.27. Top of the list is 的.)*

We handed the question "what does 苹果 mean in this sentence" to the average of a pile of vectors —
and the word that average is most similar to is **「的」**.

What does 「的」 contribute to the meaning of 苹果 in this sentence? Nothing.
On what grounds does it get 1/10? On the grounds that it appears in this sentence. That's all.

From another angle: take the words out of the sentence one at a time and see how much the context vector moves:

```text
    拿掉的词    变化量
    ----------------------
    我          0.221
    昨天        0.290
    在          0.310
    商场        0.317
    看到        0.223
    刚刚        0.217
    发布        0.314
    的          0.237
    新          0.311
    手机        0.247
```

*(Removing each word in turn and measuring how far the context vector moves: 我 "I" 0.221, 昨天 "yesterday" 0.290, 在 "at" 0.310, 商场 "mall" 0.317, 看到 "saw" 0.223, 刚刚 "just" 0.217, 发布 "release" 0.314, 的 (the possessive particle) 0.237, 新 "new" 0.311, 手机 "phone" 0.247.)*

If this representation really had read the sentence, then:

- take away 「我」 and 「的」, and the meaning of 苹果 should **hardly change**;
- take away 「发布」 or 「手机」, and the meaning of 苹果 should **clearly change**.

The actual results are 0.221 and 0.314 — about the same. All ten numbers are crammed between 0.2 and 0.3, the largest less than twice the smallest. **Averaging is equally sensitive to every word, which is to say it cannot tell one word's weight from another's.**

(Which incidentally explains why the question "should we look at the whole sentence or at a window" has no answer.
`after.py` tries taking them differently: not the whole sentence, only the two words to either side of 苹果.
The vector averaged out of `商场 看到 刚刚 发布` and the vector averaged out of the whole sentence have a cosine similarity of only +0.5148 — a gap of nearly half. Which one is right? There's no basis for saying.
**Changing the window only changes the scope of "treating everything alike"; it doesn't solve "treating everything alike" itself.**)

And so the question becomes very concrete:

> In 「我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机」,
> 「发布」 and 「昨天」 plainly contribute different amounts to the meaning of the word 苹果.
> Yet averaging hands each of them 0.10, exactly the same.
>
> So **who gets to set these weights?**

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Change `.mean(axis=0)` in `context_vector` to `.sum(axis=0)` and run it again. Which conclusions still hold, and which stop holding?
2. Change `context_vector` so that it only looks at one word to either side (window = 1), then run the little task from experiment two of `experiment.py`. Does it still get all 5 sentences right? What does the gap become for the 5th sentence (the long one)?
3. Compute the similarity between the context vector of 苹果 in the long sentence and every word in the sentence, and sort them from high to low. Who comes first? If you sorted by "how important this word is to the meaning of 苹果 in this sentence", how would you sort them? Do the two orderings agree?
4. In the long sentence, change the 苹果 at position 5 into 香蕉, into 小王 ("Xiao Wang", a person's name), into 的 — how much does the context vector differ? How much of the information "which word this is" is left in the representation? When is this "doesn't look at itself" property an advantage, and when is it a fatal flaw?
