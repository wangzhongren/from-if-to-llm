**English** | [中文](exercises.md)

# Chapter 21 Exercises

Change the code in `after.py` (or `experiment.py`), run it again, and see whether the output changes. Each exercise gives you a conclusion you can verify yourself.

---

## 1. What if you block the diagonal too

This line in `causal_mask`:

```python
return np.triu(np.ones((length, length), dtype=bool), k=1)
```

Change `k=1` to `k=0` and re-run `after.py`. Note: the entire row for position 0 is now `X`.

Guess first what position 0 will see, then check it by hand. Here's how: print the attention weights —

```python
weights = attention.masked_fill(causal_mask(length), -1e9).softmax(axis=-1)
print(np.round(weights.data[0][0], 4))
```

- What do the weights in position 0's row add up to? What is each one? Which positions did it see?
- Once that's clear, explain: why did `-1e9` **not** do its job this time? (Hint: look at the first line of `softmax` in `toygrad`.)
- What does the loss become in the causal-mask group? Is the continued text still right?

---

## 2. How negative does -1e9 actually need to be

In `after.py`, change

```python
attention = attention.masked_fill(causal_mask(length), -1e9)
```

to `-1e1`, to `-10`, to `-1`, to `0`, running `after.py` each time, and watch the loss and the continued text in the causal-mask setting.

- Which values can still train normally?
- Write a few lines of code to print the weights at those positions after the softmax, and see whether they are "very small positive numbers" or "exactly 0".
- Why is -1e9 safe and -10 not? (Hint: how big are the attention scores themselves? What order of magnitude?)

---

## 3. Opening it up one slot at a time

`experiment.py` has:

```python
LOOKAHEADS = (0, 1, 3, 7, 999)
```

Change it to `(0, 1, 2, 3, 4, 5, 6, 7, 8)` and re-run.

- From which slot onward does the "loss after blocking the future" go above 1?
- What does that threshold tell you? — Hint: when position i can see i+k, is the i+1 it has to predict inside the visible range?
- Is there a setting where it "can see the future but hasn't learned to copy"? If you don't find one, set the training steps to 400 and try again, to see whether copying takes a while to learn.

---

## 4. How big is the mask matrix

The mask printed right now is 8×8. Change that line in `after.py` to `causal_mask(32)` and see how big it gets.

- How many `X` are there in a 32×32? (Hint: `32*31/2`.)
- If the context is 1000, how many numbers does this matrix have? And if the context is 128,000 (the scale of real models today)?
- This matrix is **not** a parameter (it doesn't need to be learned), but it has to exist on every forward pass. Is that the same thing as the "parameter count" you computed in Chapter 20's Exercise 4? Why or why not?

---

## 5. Using the mask in the wrong place

The line `attention = attention.masked_fill(...)` acts on the scores **before** the softmax.

Try moving it to **after** the softmax:

```python
weights = attention.softmax(axis=-1)
weights = weights.masked_fill(causal_mask(length), 0.0)   # 注意这里填的是 0
attended = weights @ value
```

*(Note that here you fill in 0.)*

Run `after.py`.

- Does training still work? Is the continued text still right?
- Why fill in `0` here, when before the softmax you fill in `-1e9`? (Hint: what are those numbers before the softmax called? In Chapter 5 we called them…)
- Which of the two approaches is less trouble? Why can't the less troublesome one be used before the softmax?

---

## 6. Answer in one sentence

No code needed.

What this chapter taught is "a low training loss does not mean the model learned what you wanted it to learn".

Now imagine a situation: you're training a model and it reports a very low loss. The only number you have is the loss. **Without redesigning the experiment, what could you think of to judge whether it's copying?**

(Once you've thought about it, go look at the "measure it again after blocking the future" approach in `experiment.py`. Is your idea similar to it?)
