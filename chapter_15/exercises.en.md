**English** | [中文](exercises.md)

# Chapter 15 Exercises

Change things in `after.py`, run it again, and see whether the output changes.

---

## 1. Keep only one head

Set `NUM_HEADS` to 1:

```python
NUM_HEADS = 1
```

Run it once, then answer:

- How many rows of weights does the `他` ("he") position have now?
- What is its maximum? How does it compare with the 4-head case?
- Can the loss still come down low? (Yes. There are only two sentences of data; one head
  can memorize them too.)
- So why do we still want 4 heads?

---

## 2. Open more heads

Set `NUM_HEADS` to 8 or 12 and run it once.

- What does each head's dimension become? (`DIM // NUM_HEADS`)
- With more heads, can you still read the weight matrices? Is each head still attending
  to the same things?
- What happens if you set `NUM_HEADS` to 5? (`24 // 5 = 4` — where did the remaining 4
  dimensions go?) This is why `DIM` has to be divisible by the number of heads.

---

## 3. Change the seed

Change `train(steps=400)` to `train(steps=400, seed=7)`, and try a few more seeds.

- Is the `他` row of the 4 heads different every time?
- Is there any head that looks at the same word **every** time?
- Think one thing through: with only two sentences of data, we **cannot** say "head 1
  handles reference". So what is it that we *can* say for certain?

---

## 4. Make several heads share one set of Q/K/V

In `multi_head_attention`, change the three lines to use only head 0's projections:

```python
query = vectors @ params["w_q"][0:1]
key = vectors @ params["w_k"][0:1]
value = vectors @ params["w_v"][0:1]
```

Run it again. Are the 4 heads' weight matrices the same now?

(This is the meaning of the name "multi-head": **what makes the heads different is that
each has its own projection** — it isn't just computing the same thing several times.)

---

## 5. Replace concatenation with addition

The last line of `multi_head_attention` does "concatenate, then project". Try turning it
into "add":

```python
joined = head_output.sum(axis=0)      # (词数, 每个头的维度)
```

*(The comment says `(词数, 每个头的维度)` — "(number of words, per-head dimension)".)*

Do the shapes line up? What error does it throw when you run it? If you wanted the shapes
to line up, what shape would `w_o` have to become? And think: which keeps more
information, concatenation or addition?

---

## 6. A tiny bit more data

The training data is two sentences right now. Add the sentence from Chapter 13 (it's
already in `SENTENCES`), then try adding one of your own (**this is only for the exercise
— don't write it into the book's corpus**):

```text
小王 昨天 把 书 给了 小李
```

*(`小王 昨天 把 书 给了 小李` — "Xiao Wang gave the book to Xiao Li yesterday".)*

Train once more and see whether the `他` row changes.

What this exercise is getting at: **what multi-head can express depends on whether there
is enough data to learn it from**. With two sentences, what the heads learn is luck.
