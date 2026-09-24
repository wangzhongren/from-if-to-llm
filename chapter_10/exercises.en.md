**English** | [中文](exercises.md)

# Chapter 10 Exercises

How to run:

```bash
./.venv/bin/python chapter_10/before.py
./.venv/bin/python chapter_10/after.py
./.venv/bin/python chapter_10/experiment.py
```

---

## 1. How lucky can the numbering get

Scheme C in `before.py` shuffles the words using a fixed seed. Replace the `np.random.default_rng(0)` in `build_schemes` with other seeds, run it several times, and write down the accuracy each time.

**Question**: what range do these accuracies jump around in? Why are they all below 100%, and yet not the 50% of guessing at random either?

**Hint**: work out what a single straight line can do to these 15 words, then think about the probability that a random arrangement happens to put "all the tech words in front" (`experiment.py`'s second experiment computes that number).

---

## 2. How many numbers is "a list of numbers"

`DIM = 16` in `after.py` is how many numbers each word gets. Change it to 2, 4 and 8, and run each.

The results we got (`seed=0`):

| DIM | Accuracy under the three numberings |
|---|---|
| 2 | 53.3% |
| 3 | 73.3% |
| 4 | 66.7% |
| 8 | 80.0% |
| 16 | 100.0% |

**Question**: with fewer numbers, why do the vectors also fail to separate the words? Is that the same reason as "the IDs can't separate them"?

**Hint**: put 15 points into a 2-dimensional plane and it is not always possible to split them into two piles with a straight line in any way you like. The fewer the dimensions, the fewer "splits" fit inside — and that is two versions of one and the same thing as "an ID is too poor, it can't hold meaning" from Chapter 10.

---

## 3. Give every ID an offset of its own

The model in `before.py` is:

```python
score = w * token_id + b
```

Now give every word an offset of its own (that is, 15 `b_id`s):

```python
score = w * token_id + b_id
```

**Question one**: with that change, can scheme B (pinyin order) learn? Try it.
**Question two**: what are those 15 `b_id`s? How do they differ from "a list of numbers per word"?
And how do they compare with `DIM = 1` in `after.py`?

**Hint**: if you notice that "this is just swapping the ID for a table lookup" — right, and that is where this road ends when you walk it all the way.

---

## 4. "Similarity" among random vectors is luck

`after.py` prints these two lines:

```text
和 手机   最像的 5 个词： 派 +0.28  这个 +0.15  电脑 +0.13  做成 +0.09  小米 +0.09
和 苹果   最像的 5 个词： 华为 +0.36  做成 +0.25  发布 +0.24  很 +0.21  小米 +0.19
```

*(Cosine similarity, top 5 words: the five words nearest 手机 "phone" are 派 pie +0.28, 这个 this +0.15, 电脑 computer +0.13, 做成 make-into +0.09, 小米 Xiaomi +0.09; the five nearest 苹果 "apple" are 华为 Huawei +0.36, 做成 +0.25, 发布 release +0.24, 很 very +0.21, 小米 +0.19.)*

Change `SEED` in `after.py` to 1, 2, 3, 4 and 5, run each, and write down "the word most like 手机".

**Question**: what five answers do the five seeds give? Is there any pattern in them?
If someone says "you can see a bit of meaning in the random vectors too — look, 手机 and 小米 are sort of alike", how would you answer them?

---

## 5. Build a numbering scheme of your own

The second experiment in `experiment.py` says that among the 15 words, only two arrangements can be separated by a straight line: "all tech words in front" or "all food words in front".

**Task**: without changing the model and without changing the words, only the ID order, try to construct a numbering scheme that is **less accurate than scheme A but more accurate than scheme B**.

**Question**: does such a scheme exist? If it does, what do its IDs look like?
If it doesn't, why not?

**Hint**: a straight line makes exactly one cut. Think about "where that one cut has to fall for the most words to land on the correct side".
