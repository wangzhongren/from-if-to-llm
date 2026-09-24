**English** | [中文](exercises.md)

# Chapter 5 exercises

The code in this chapter is absurdly short: `softmax` is six lines, `loss_of` is three.
But it's the first "real" mathematics in the book — **intuition first, formula second.**

So half of these exercises are by hand. Don't skip them; you only find out what the formula is
saying once you've worked through it.

---

## 1. Hand-compute one softmax and one loss

No running the program. Work out the probability and the loss for 「苹果芯片很强」 under table A
with pen and paper.

Table A gives it two scores:

```text
科技 1.0    食品 -1.0
```

*(technology 1.0, food -1.0.)*

Step one, compute `e to the power of 1.0` and `e to the power of -1.0`:

```text
e^1  ≈ 2.7183
e^-1 ≈ 0.3679
```

*(e^1 is about 2.7183, e^-1 is about 0.3679.)*

Step two, add them up and divide by the total:

```text
科技的概率 = 2.7183 / (2.7183 + 0.3679) = 0.8808
```

*(the probability of technology = 2.7183 / (2.7183 + 0.3679) = 0.8808)*

Step three, compute the loss:

```text
损失 = -log(0.8808) = 0.127
```

*(loss = -log(0.8808) = 0.127)*

When you're done, run `experiment.py` and find this row in experiment 2's table:

```text
  苹果芯片很强           2.0      1.00     0.127   只赢一点点，最勉强
```

*(苹果芯片很强, won by 2.0, square error 1.00, cross-entropy 0.127, "wins by a tiny bit, the most
reluctant".)*

Does it match?

---

## 2. How much to change each time

Change this in `after.py`:

```python
ADJUST_AMOUNT = 0.5
```

to `0.05` and run it. Then change it to `5.0` and run it again.

```text
  每次改多少           轮数          最终损失     正确率
  0.05           229      8.62e-11    10/10
  0.5             26      1.00e-11    10/10
  2.0              8      2.62e-12    10/10
  5.0              5      4.69e-14    10/10
```

*(Columns: how much we change each time, rounds, final loss, accuracy. 0.05: 229 rounds, 8.62e-11,
10/10. 0.5: 26 rounds, 1.00e-11, 10/10. 2.0: 8 rounds, 2.62e-12, 10/10. 5.0: 5 rounds, 4.69e-14,
10/10.)*

Think it through:

- Why do smaller changes need more rounds? (Hint: how far does one step of 0.05 travel compared
  with one step of 5.0?)
- In Chapter 4 we ran the same experiment and all four step sizes produced **identical** training
  trajectories. Why is it different in this chapter?

---

## 3. Swap in a different ruler (the result will make you blink)

Change `loss_of` to use an "error rate" as the loss:

```python
def loss_of(words, label):
    correct_index = CATEGORIES.index(label)
    return 1.0 - probability_of(words, correct_index)
```

Run `after.py`.

```text
轮数：26        最终损失：1.003e-11
```

*(Rounds: 26. Final loss: 1.003e-11.)*

**Exactly the same as with `-log`.** Even the number of rounds matches.

Why? Think it through: when there are only two categories, what does `1 - the probability of the
correct answer` equal?

(Answer: it equals **the probability of the wrong category**. And "the wrong probability is
smaller" and "the correct probability is larger" are the same thing. The two rulers have different
marks, but the **order** between the marks is the same — so "which is better" comes out identical
every time.)

Which shows one thing: in this "try one parameter at a time" approach, the ruler's order matters
more than the ruler's values.

---

## 4. Take out softmax's guard

Two lines in `softmax` look a bit redundant:

```python
biggest = max(scores)
exponentials = [math.exp(value - biggest) for value in scores]
```

Try changing them to the plainest version:

```python
exponentials = [math.exp(value) for value in scores]
```

Then run this:

```python
print(softmax([1000.0, 0.0]))
```

```text
OverflowError: math range error
```

*(OverflowError: math range error.)*

Why? How big is `e` to the 1000th power? Write it down and count the digits.

Then think one step further: after subtracting the maximum, the largest `e` exponent is 0, which
means nothing exceeds 1 — **that is the entire job those two lines do**. Does the result change?
(No, because numerator and denominator shrink together.)

---

## 5. Measure yourself with the ruler

Suppose you have three models, and this is how they do on the 10 sentences:

```text
  模型 A：10 句全对，平均损失 0.0000
  模型 B：10 句全对，平均损失 0.2100
  模型 C： 9 句全对，平均损失 0.1500
```

*(Model A: all 10 sentences right, average loss 0.0000. Model B: all 10 right, average loss 0.2100.
Model C: 9 right, average loss 0.1500.)*

What would Chapter 4's ruler (how many were answered wrong) say? Which would it think is best?

What would this chapter's ruler say? Which would it think is best?

If you had to ship one model, which would you pick? And why might "C, with 9 right" be worth a
second look over "B, with all 10 right"?

---

## 6. Challenge: why can't we move all the parameters at once?

Right now `adjust_weights_once` moves one parameter at a time, and after each move it recomputes
the entire corpus.

Try changing it to "move all the parameters at once": nudge every parameter in some direction,
then compute the loss once.

You'll run into these problems:

- Which direction should each parameter move in? What information do you have? (Hint: you have
  exactly one loss value.)
- If the loss goes up after changing everything, how do you know **which** parameter broke it?
- If the loss goes down after changing everything, how do you know that some parameter didn't
  break it while others improved so much that they covered up the damage?

Those three questions are the entire content of the next chapter.
