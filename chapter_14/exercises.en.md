**English** | [中文](exercises.md)

# Chapter 14 Exercises

Change the functions in `after.py`, run it again, and see whether the output changes.

---

## 1. What if the three matrices become one

Try setting the three projections to the same matrix:

```python
w_q = randn(DIM, D_KEY, seed=14)
w_k = w_q
w_v = w_q
```

Run it once, then answer:

- Are the scores for `苹果 -> 手机` and `手机 -> 苹果` (`apple -> phone`, `phone -> apple`)
  still different?
- Does this chapter's weight matrix look like Chapter 13's table?
- Which of the "three roles" can be dropped, and which one cannot?

(Try once more with all three set to the identity matrix `np.eye(DIM)`, and see what
happens.)

---

## 2. Take the √d away

Change `attention_scores` back to no scaling:

```python
def attention_scores(query, key):
    return query @ key.T
```

Then move `D_KEY` from 8 to 64, then to 128, and run it three times.

- In the `苹果` (apple) row, what is the largest weight? And the smallest?
- Once the weights have turned into "a 1 and a pile of 0s", is there any gradient left in
  that row? (Copy the `sensitivity` function out of `experiment.py` and compute it.)

---

## 3. Scale only, don't switch V

Change the last line of `attention` to mix the original vectors:

```python
return weights @ value, weights      # 改成
return weights @ vectors, weights    # 这样
```

*(The comments say `改成` "change to" and `这样` "like this".)*

Run it once. Is the shape of the output still the same? How far off are the numbers?

And think: the weights decide "who to look at", the value decides "what you see when you
look". If the two are not separated at all, does Chapter 13's defect come back?

---

## 4. Invent a question of your own

In the last section, change the dimension being asked about to dimension 1 (time):

```python
for dim in [5, 4, 1]:
```

- Who does `苹果` go and ask now?
- Can it get anything out of them? Why? (Hint: look at what `苹果`'s vector has in the
  time dimension.)
- If a word simply doesn't carry that dimension's information on it, what will the
  weights it asks with look like?

What this exercise shows is: the query is "what I'm looking for", but if **I haven't got
that thing on me**, I can't even ask the question.

---

## 5. Make the question ask harder

Change `QUESTION_STRENGTH` from 4.0 to 0.1, then to 20.0, and run it each time.

- Do the weights flatten out or sharpen?
- Who decides this "strength"? (Remember `before.py`'s "changing 苹果's own vector also
  changed how 手机 sees 苹果"? Now that we change the strength of the query, does it
  affect the key?)

---

## 6. The exam question of another sentence

Think about the sentence Chapter 15 is going to use (no code needed yet):

```text
小王 把 书 给了 小李 因为 他 明天 考试
```

*("Xiao Wang gave the book to Xiao Li because he has an exam tomorrow." Word by word:
`小王` Xiao Wang · `把` the *bǎ* particle marking the thing acted on · `书` book · `给了`
gave · `小李` Xiao Li · `因为` because · `他` he · `明天` tomorrow · `考试` exam.)*

If we let `他` ("he") ask about **reference** (who is "he"?), which words should it attend
to? If we let it ask about **time** (when is his exam?), which words should it attend to
then?

Both questions would have to be satisfied by the same row of weights — write down the
ideal weights for each, and see whether they add up to more than 1.
