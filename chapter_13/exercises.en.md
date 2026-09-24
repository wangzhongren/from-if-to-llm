**English** | [中文](exercises.md)

# Chapter 13 Exercises

Change the functions in `after.py`, run it again, and see whether the output changes.
Every exercise gives you a conclusion you can verify yourself.

---

## 1. What happens without normalization

Change `attention_weights` to "return the scores directly" — that is, skip the softmax:

```python
def attention_weights(scores):
    return scores
```

Run `after.py` once.

- Look at the `苹果` (apple) row: do the weights still sum to 1?
- Look at the new vector: has its length grown or shrunk?
- Which words' weights turn into 0 or negative numbers?

Then put softmax back, but remove only the "subtract the maximum" step (hint: what
happens when the scores are very large and you *don't* subtract the maximum?). Think
about why this step doesn't change the result, yet has to stay.

---

## 2. Make softmax "hard"

Replace `attention_weights` with "give 1 to the highest-scoring word only, 0 to
everything else" (you can find it with `np.argmax` first).

Run it, then answer:

- What does `苹果` end up becoming the vector of? Can it still carry the information
  from both `发布` (released) and `手机` (phone) at once?
- Compared with "averaging", is this the opposite extreme?

(Hint: this is what people years later called "hard attention". Its flaw is that you
can't get a gradient out of it — Chapter 9's `backward()` can't walk through it.)

---

## 3. A different kind of "alike"

Swap `similarity_scores` from the dot product to cosine similarity:

```python
def similarity_scores(vectors):
    x = vectors.data
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return Tensor(x) @ Tensor(x).T
```

Run it once.

- Did the gaps between the weights grow or shrink? Why?
- What is `商场` ("the mall")'s weight now? How much did it change from before?
- Between the dot product and cosine, which one cares more about "same direction", and
  which one cares more about "length"?

---

## 4. One extra word in the sentence

Add a word to `EMBEDDINGS`, for example:

```python
"香蕉": [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
```

Change the sentence to `我 昨天 在 商场 看到 苹果 刚刚 发布 的 新 手机 香蕉`
("... phone banana"), and run it once.

- Under averaging, what do the other 10 words' weights change from `1/10` to?
- Under attention, how much does the weight `苹果` gives `手机` drop?
- Adding one completely irrelevant word to a sentence — does it hurt averaging more,
  or attention more?

---

## 5. Make the "looker" into somebody else

Right now row i of the weight matrix is "who word i is looking at". Change `苹果`'s
vector (say, change the "tech & business" dimension from 1.0 to 0.0) and run it again.

- Did who `苹果` looks at change?
- Did the weights with which others look at `苹果` change?

If the answers to these two questions are the same, then in the current scheme "what
I'm looking for" and "what I have" are one and the same thing — which is exactly what
the next chapter is going to take apart.
