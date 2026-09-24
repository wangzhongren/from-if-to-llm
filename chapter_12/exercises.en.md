**English** | [中文](exercises.md)

# Chapter 12 Exercises

How to run:

```bash
./.venv/bin/python chapter_12/before.py
./.venv/bin/python chapter_12/after.py
./.venv/bin/python chapter_12/experiment.py
```

---

## 1. Mean, or sum

`context_vector` uses `.mean(axis=0)`. Change it to `.sum(axis=0)` and run it again.

**Question one**: does the accuracy of the little task from experiment two of `experiment.py` change?
(What we got: all 5 sentences still correct, and not one number of the gap changed.)

**Question two**: why does switching to a sum change nothing at all?

**Question three**: what about the "take a word out and see how far the vector moves" table in `after.py`? With the sum version, how far does the vector move when you take out 「发布」 "release", and when you take out 「我」 "I"? Are they still "about the same"?

**Hint**: cosine similarity looks only at direction, never at length. And the summed vector is the averaged vector times 10 — an identical direction. But "take one word out" measures a change in **length**, and that quantity is a completely different thing in the two versions.

---

## 2. How wide a window

`context_vector` currently uses the whole sentence. Change it to look only one word to either side:

```python
left = max(0, position - 1)
right = min(len(sentence), position + 2)
window = [ids[i] for i in range(left, right) if i != position]
return table_data[window].mean(axis=0)
```

Then run experiment two of `experiment.py`.

The results we got:

| Sentence | Gap with the whole sentence averaged | Gap with window = 1 |
|---|---|---|
| 苹果 很 好吃 (apples are delicious) | +0.634 | +0.520 |
| 苹果 很 甜 (apples are very sweet) | +0.666 | +0.520 |
| 苹果 发布 新 手机 (Apple shipped a new phone) | −0.729 | −0.245 |
| 苹果 发布 新 芯片 (Apple shipped a new chip) | −0.639 | −0.245 |
| 我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 (I saw the new phone Apple just released at the mall yesterday) | −0.696 | **−0.058** |

**Questions**:
- All 5 sentences are still correct. But the gap for the last one fell from −0.696 to −0.058 — what does that mean?
- Should the window be 1 word, 2 words, or the whole sentence? Can you say clearly which one is "right"?

**Hint**: −0.058 is very nearly a tie. Change the random seed or add one sentence and it could flip over to the positive side. **An answer that is only right by luck doesn't count as right.**

---

## 3. Which word in the sentence is the context vector most like

In the long sentence `我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机` ("I saw the new phone Apple just released at the mall yesterday"), compute the similarity between 苹果's context vector and **every word in the sentence**, and sort from high to low.

The ordering we got:

```text
   的     +0.477
   新     +0.377
   手机   +0.305
   看到   +0.273
   发布   +0.267
   刚刚   +0.193
   商场   +0.049
   在     +0.047
   我     -0.056
   昨天   -0.109
```

*(The words, from most to least similar to the context vector of 苹果: 的 (the possessive particle) +0.477, 新 "new" +0.377, 手机 "phone" +0.305, 看到 "saw" +0.273, 发布 "release" +0.267, 刚刚 "just" +0.193, 商场 "mall" +0.049, 在 "at" +0.047, 我 "I" −0.056, 昨天 "yesterday" −0.109.)*

**Question one**: the top of the list is 「的」, the possessive particle. If you sorted by "how important this word is to the meaning of 苹果 in this sentence", where would you put 「的」? And 「发布」?

**Question two**: the last two are 「我」 "I" and 「昨天」 "yesterday". Are they really unimportant? Change 「昨天」 to 「明天」 "tomorrow" — does the meaning of 苹果 change?

**Question three**: was this ranking computed by the averaging scheme itself, or did we lay it down by hand?

---

## 4. Swap the word, get an identical representation

This chapter's scheme is:

```python
context(i) = 周围所有词的向量的平均
```

*(context(i) = the average of the vectors of all the words around it.)*

**Note that the expression contains no "the i-th word itself".**

**Task one**: in the long sentence `我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机`, replace the 苹果 at position 5 with 香蕉 "banana". How far does the resulting context vector differ from the original?

The result we got: 0.0000000 — not one number differs.

**Task two**: change 苹果 into 小王 ("Xiao Wang", a person's name), into 的, into 商场 "mall" — the result is the same every time.

**Questions**:
- Which word this "苹果" in the sentence actually is — how much of that information is left in the context vector?
- Then where else can the model learn "which word this is"?
- When is this "doesn't look at itself" property an advantage, and when is it a fatal flaw?

**Hint**: go to `after.py` first and read those three lines of `context_vector`. Look at where the `position` argument is actually used.

---

## 5. How many meanings can one word have

This chapter turned "苹果's representation" from 1 into "1 per position".
Count how many times 苹果 appears in the corpus, then answer:

**Question one**: among these 33 words, which word has the most representations, and how many does it have?

**Question two**: if the corpus were a book, how many representations would that word have? Storing every position's representation — how much space would that take up?

**Question three**: no real system can store one vector per position. So how do they deal with it?
(Keep this question with you as you read on; Chapter 13 will offer one line of thinking.)
