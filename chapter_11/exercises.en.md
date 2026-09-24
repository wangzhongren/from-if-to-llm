**English** | [中文](exercises.md)

# Chapter 11 Exercises

How to run:

```bash
./.venv/bin/python chapter_11/before.py
./.venv/bin/python chapter_11/after.py
./.venv/bin/python chapter_11/experiment.py
```

---

## 1. How far does a neighbour look

`WINDOW = 1` in `after.py` decides how far "guess the neighbour" looks. Change it to 2 and 3 and run each.

The results we got:

| WINDOW | Training samples | loss | Within − between |
|---|---|---|---|
| 1 | 68 | 1.0748 | +0.576 |
| 2 | 110 | 1.6328 | +1.049 |
| 3 | 126 | 1.8925 | +1.270 |

**Question**: with a bigger window there are more training samples, yet the loss goes **up** — while "within − between" goes **up** too. Are those two things in conflict?

**Hint**: the loss measures "how accurately the guess lands". With a bigger window each sample has more neighbours to guess, and 新 "new" may have 发布 "release" on its left and 手机 "phone" on its right — more than one word can stand beside a given word in the first place, so guessing wrong is normal. So what does "within − between" measure? It measures whether the picture that ends up being drawn has groups in it. The two things measure different things, so one can get worse while the other gets better.

**Think further**: with the window opened to 3 (which covers the whole sentence), how much information is left in the "guess the neighbour" task? If the window were as long as the sentence, what would go wrong?

---

## 2. How many numbers per word

`DIM = 8` in `after.py` is how many numbers each word gets. Change it to 2, 4 and 16, run each, and write down the "within − between" gap from experiment two of `experiment.py`.

The results we got:

| DIM | Within − between |
|---|---|
| 2 | −0.127 |
| 4 | +0.130 |
| 8 | +0.576 |
| 16 | +0.452 |

**Question**: why don't 2 numbers work, and why is 16 no better than 8?

**Hint**:
- With 2 numbers, all 16 words are crammed into one plane. For "the tech words clump together and the food words clump together" to hold at the same time, two directions simply aren't enough — the same kind of dead end as "the IDs can't separate the words" in Chapter 10.
- With 16 numbers, the model is capable of memorising all 68 training samples one by one. Once they're memorised the loss is low, but what it learned is "the answers to these 68 samples", not "the relations between the words". Go and look: the loss at DIM=16 (1.05) is even lower than at DIM=8 (1.07) — and yet its structure is blurrier.

---

## 3. Make 苹果 move

In the original corpus, 苹果 appears in 3 tech sentences and 4 food sentences, so it leans slightly towards the food side.

**Task one**: add these three sentences to the corpus —

```python
"苹果 发布 新 手机",
"苹果 发布 新 芯片",
"苹果 发布 新 电脑",
```

and train again. Compare 苹果's neighbour list with the original.

The results we got:

```text
原始       苹果→ 香蕉 0.82  华为 0.61  好吃 0.61  小米 0.61  甜 0.56
科技句 ×2  苹果→ 香蕉 0.75  华为 0.72  小米 0.72  好吃 0.58  甜 0.53
```

*(苹果's five nearest words before the change, and after adding the two tech sentences. The tech neighbours 华为 "Huawei" and 小米 "Xiaomi" climb from 0.61 to 0.72, while the food neighbours 好吃 "delicious" and 甜 "sweet" slip from 0.61 and 0.56 to 0.58 and 0.53 — 苹果 has moved towards the tech side.)*

**Task two**: replace the sentences you added with food sentences (say `"苹果 很 甜"` — "apples are very sweet") and try again.

**Question**: why does 苹果's vector move along with the recipe of the corpus? Does it "know" that it has moved?

---

## 4. Distinctions the corpus doesn't show

In the corpus, 华为 and 小米 both appear only in the pattern `"X 发布 新 Y"`, and can occupy exactly the same positions.

**Task**: find the 华为–小米 cell in the similarity matrix (it is 0.99). Now find the 华为–手机 cell (it is −0.03).

**Questions**:
- Why are 华为 and 小米 almost the same vector?
- Does the model know that "华为 and 小米 are two different companies"? Could it possibly know?
- If we want the model to tell them apart, what is the least we have to add to the corpus?

**Hint**: the model can only learn differences that appear in the corpus.
This is not just a "quirk of the toy model" — the real thing behaves the same way: for anything the corpus doesn't spell out, it can only guess.

---

## 5. Guess in one direction only

The training samples right now are **two-directional**: for two neighbouring words, both directions are collected. For instance 「苹果 发布」 produces both "see 苹果, guess 发布" and "see 发布, guess 苹果".

**Task one**: swap the `seen` and `guessed` arguments of `train_embeddings(seen, guessed)` in `after.py` and train again.

The result is identical (the loss is still 1.0748, and not one number in the neighbour table changes). Work out why — once that clicks, you understand what two-directional sampling means.

**Task two**: collect only one direction. Replace `build_training_pairs` with this:

```python
for i in range(len(sentence) - 1):
    seen.append(sentence[i])
    guessed.append(sentence[i + 1])     # 只用「左边的词」猜「右边那个词」
```

The samples drop from 68 to 34. The results we got:

```text
loss = 0.6919      （比双向的 1.0748 低 —— 任务本身变简单了）
手机 → 华为 0.50  小米 0.49  做成 0.42  芯片 0.41
好吃 → 芯片 0.48  派 0.39  苹果 0.30  华为 0.19
```

*(One-direction training. The loss is 0.6919 — lower than the 1.0748 of the two-direction version, because the task itself got simpler — but 手机 "phone"'s nearest words are now 华为 "Huawei" 0.50, 小米 "Xiaomi" 0.49, 做成 "make-into" 0.42 and 芯片 "chip" 0.41, and the word most like 好吃 "delicious" has become 芯片 "chip" 0.48.)*

**Question**: the loss is lower, yet the structure has collapsed: the word most like 好吃 has become 芯片.
Why does losing half the samples make that much difference?

**Hint**: with two-directional sampling, every word gets a chance to appear as "the answer being guessed" and also as "the clue used to guess". Keep only one direction and a word is trained on only one of those sides. Think about what that means for its vector.

**Think further**: if you kept only one direction but made the corpus a hundred times bigger, would the structure grow back?
