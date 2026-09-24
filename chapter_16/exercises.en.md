**English** | [中文](exercises.md)

# Chapter 16 Exercises

Change things in `after.py` and `experiment.py`, run them again, and see whether the
output changes.

---

## 1. How much positional encoding is right

In `after.py` the position vector is **added** straight on:

```python
vector = Tensor(word_vectors + position_table)
```

Try multiplying it by a factor first:

```python
vector = Tensor(word_vectors + 0.1 * position_table)
vector = Tensor(word_vectors + 10.0 * position_table)
```

Run `experiment.py` each time and look at:

- What does the two sentences' sentence-vector distance become?
- When the factor is very large, do the input vectors still look like the original words?
  (Work it out with the method from the third table in `experiment.py`.)
- How loud does the position information have to be before it can be heard? And how much
  of the word's meaning gets covered up in exchange?

---

## 2. Drop sin/cos for random vectors

The positional encoding doesn't have to use sine and cosine — can't we just give each
position a fixed random vector? Try it:

```python
def random_positions(length, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((length, dim))
```

Use this table to run experiment one in `experiment.py`.

- Can the two sentences be told apart now? (Yes.)
- Does the "similarity depends only on distance" table from experiment two still hold?
  (It doesn't.)
- Think about why "similarity depends only on distance" is useful for language. (Hint: the
  relation between a word and the word right before it is not the same as the relation
  between it and the word two before it; and "one word apart" ought to come out about the
  same wherever in the sentence it happens.)

---

## 3. Where the position vector goes

The current scheme is "add". Try switching to "concatenate":

```python
vector = Tensor(np.concatenate([word_vectors, position_table], axis=-1))
```

If you do that, what does `DIM` have to become? Do the shapes of `w_q`, `w_k`, `w_v` have
to change along with it? Does it run?

(Add and concatenate are both schemes that really existed. Adding keeps the dimension
unchanged, at the cost of mixing position and word meaning together; concatenating keeps
both sides intact, at the cost of doubling the dimension.)

---

## 4. A longer sentence

Change `LENGTH` from 11 to 100 and run experiment three in `experiment.py`.

- What is the positional encoding's length at position 99? And its value range?
- Are any of the position vectors duplicates? (Hint: look at the similarity in the
  distance-0 row.)
- Training only ever saw 11 positions, and then 100 show up out of nowhere — how well does
  the positional encoding "extrapolate"?

---

## 5. Which sentence got shuffled

"狗咬人" and "人咬狗" are **a whole sentence flipped around**. Try swapping just two words:

```python
SENTENCE_C = "人 咬 狗".split()      # 已知
SENTENCE_D = "狗 人 咬".split()      # 只把后两个词换了一下
```

*(`已知` — "already known"; `只把后两个词换了一下` — "only the last two words were swapped".
So `SENTENCE_C` is "man bites dog" and `SENTENCE_D` is "dog person bites".)*

What is the distance between the two sentences' output vectors? And compared with
flipping the whole thing around?

What this exercise is getting at: a positional encoding remembers **the thing at each
position**, not "whether this sentence is backwards". Being backwards is just one case it
happens to be able to express along the way.
