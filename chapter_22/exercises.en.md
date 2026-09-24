**English** | [中文](exercises.md)

# Chapter 22 Exercises

Change the constants in `after.py`, run it again, and see whether the output changes. Each exercise gives you a conclusion you can verify yourself.

---

## 1. Add layers until it hurts

`after.py` has:

```python
LAYERS = 3        # 堆几个 Transformer Block
```

*(How many Transformer blocks to stack.)*

Change it to 6, then 12, re-running each time.

- What are the parameter counts?
- Does the loss get better? (On my run 3 layers gave 0.0349, and 6 and 12 layers were worse.)
- If it got worse, is that "too many layers are bad"? Or something else?
  Hint: think about how many parameters this model has in total, and how many characters the corpus has in total.

---

## 2. The number of heads

`HEADS = 4`, `DIM = 64`, so each head gets `head_dim = 16`.

Change `HEADS` to 1, then to 8 and 16, running each one.

- Does the parameter count change? (Hint: does the shape of `query_weight` have anything to do with `HEADS`?)
- Does the loss change?
- With `HEADS = 16`, what is `head_dim`? Does it still work?
- If someone tells you "multi-head attention makes the parameter count bigger", what do you say back?

---

## 3. Take the residual out

`TransformerBlock.__call__` has these two lines:

```python
x = x + self.multi_head_attention(normed, mask)
...
x = x + hidden @ self.proj_weight + self.proj_bias
```

Change both `x = x + ...` to `x = ...` and re-run `after.py`.

- What does the loss become?
- Is the poem it writes still right?
- Now change the layer count to 6 and try again. Why might this change be tolerable at a small number of layers and completely hopeless once there are more?

---

## 4. Don't always pick the largest

This line in `continue_text`:

```python
ids.append(int(logits.data[0, -1].argmax()))
```

Change it to sampling by probability:

```python
probabilities = logits.softmax(axis=-1).data[0, -1]
rng = np.random.default_rng(0)
ids.append(int(rng.choice(len(probabilities), p=probabilities)))
```

(Remember to create `rng` once at the top of the function, don't rebuild it every step.)

- Same opening `床前`, run it three times: are the results the same?
- Change the opening to `举头望明月低头` — is the result interesting? Or does it turn into gibberish quickly?
- Guess why. Next chapter will give you the answer.

---

## 5. Cut the context in half

`CONTEXT = 32`. Change it to 8 and 16, running `after.py` each time.

- What does the training-set accuracy become?
- Use `continue_text` to continue 60 characters — is it still right?
- Think about that curve in Chapter 22's `before.py` (window 2 to 32). Is this the same problem the "window concatenation" model ran into in Chapter 20?
- So where does attention win over concatenating windows?

---

## 6. How many times to repeat the corpus

This line at the top:

```python
CORPUS = (POEM_A + POEM_B) * 8
```

Change the `8` to `1` (so the corpus is only 38 characters) and re-run.

- How many windows can `build_batches` cut out?
- What is the loss? Is the text it writes still right?
- Now change the `1` to `80` and re-run. What about the loss?
- What do these two experiments show? (Chapter 28 will state this fact as a rule.)
