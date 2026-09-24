**English** | [中文](exercises.md)

# Chapter 20 Exercises

Change the constants in `after.py` (or `experiment.py`), run it again, and see whether the output changes. Each exercise gives you a conclusion you can verify yourself.

---

## 1. How small can the window get

`after.py` has this line:

```python
WINDOW = 8      # 往前看几个字符
```

*(How many characters to look back at.)*

Change it to 4, then 2, then 1, re-running `after.py` each time, and write down: the training-set accuracy, and the output of that 头 probe.

Hint: `predict_next` requires the prompt to have at least `WINDOW` characters, so once the window shrinks the prompt has to shrink too (for example `"霜举头望明月低头"[-WINDOW:]`).

- At which window size does the 头 probe start getting it wrong?
- When it gets it wrong, does it pick 望 "gaze" or 思 "think of"? Why that one?
- When the window is 1, what is left that distinguishes this model from the classifier in `before.py`? (Look at the parameter count.)

---

## 2. What happens if you take the embedding apart

In Chapter 10 we said "the index means nothing", so we gave each character a set of numbers that can be learned.

Now do a destructive experiment: replace the embedding table with an identity matrix (each character becomes a one-hot vector), i.e. take away the very possibility of "characters can be similar to each other".

How: in `NextCharModel.__init__`, take

```python
self.table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
```

and replace it with

```python
self.table = Tensor(np.eye(vocab_size, dim), requires_grad=False)
```

(Remember to set `DIM` to 33, or the shapes won't line up.)

- What are the loss and the accuracy? (On my run the loss went from 0.0026 up to 0.0175, and the accuracy was still 100%.)
- The accuracy didn't drop, but the loss went up. How can both of those be true at once?
- What does this change actually show? — Hint: the linear layer that follows `embedding` takes `WINDOW * DIM` inputs and produces `VOCAB_SIZE` outputs. If `table` is nothing but a fixed one-to-one mapping, then what is that linear layer equivalent to?

---

## 3. Group B's second slot becomes "the character after the answer"

In `experiment.py`, Group B's input is `[previous character, answer]` and its output is the answer. It copies the answer 72.7% of the time.

Now change Group B to `[previous character, the character after the answer]`:

```python
if second_slot == "answer":
    second = ids[position + 1]        # 改成 ids[position + 2]
```

Guess what the copy rate will become first, then run it.

- Why can't it copy this time?
- What criterion does this give you for "what counts as leakage"? Try to write it in one sentence.

---

## 4. Work out the parameter count

`after.py` prints this line:

```text
  模型参数一共 4785 个（其中 4224 个在最后那个线性层里）
```

*("4785 parameters in total, 4224 of them in that last linear layer.")*

- Where does `4224` come from? (Hint: `WINDOW * DIM * VOCAB_SIZE + VOCAB_SIZE`.)
- Change `WINDOW` to 32, keep `DIM` at 16: what's the parameter count?
- If I want it to look at 1,000 characters, where does the parameter count end up?
- That number is exactly why Chapter 21 has to replace concatenation. In Chapter 14's attention, does the parameter count have anything to do with the sequence length?

---

## 5. Swap the corpus for other text

`before.py` and `after.py` have a corpus at the top. Replace `POEM_B` with another short poem of the same length (requirement: use only characters that are already there, or handle the new characters too), and run both twice.

- Did `before.py`'s accuracy go up or down?
- Why? (Hint: in your new corpus, how many characters are followed by more than one possibility?)

---

## 6. Answer in one sentence

No code needed.

In this chapter we defined "language model": **given a stretch of text, output a probability distribution over the next token.**

Now the question: if you can only pick one of the 33 probabilities the model gives you, how would you pick? And what's the problem with always taking the largest one?

(Write your answer down first. Chapter 23 will run all three answers to this question for you.)
