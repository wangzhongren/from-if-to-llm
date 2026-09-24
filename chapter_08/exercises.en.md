**English** | [中文](exercises.md)

# Chapter 8 Exercises

## 1. Multiply that bracket out with a pen

The math is only one step, and writing it out yourself will make it stick far better:

```text
h = w1 * x + b1
分数 = w2 * h + b2
```

Substitute the first line into the second and you get
`分数 = (w2 * w1) * x + (w2 * b1 + b2)`, where `分数` is "score".

Then write out 10 layers too: `分数 = W10 * (... W2 * (W1 * x + b1) + b2 ...) + b10`.

Try to convince yourself: however many layers there are, it can always be written as
`A * x + B`.
(`A` is what that string of matrices multiplies out to, and `B` is what that string of biases
turns into.)

## 2. Swap in a different "bending" function

### (a) leaky_relu

```python
def leaky_relu(value):
    """负数不变成 0，而是变成原来的 0.01 倍。"""
    return value if value > 0 else 0.01 * value
```
*(The docstring: "negatives don't become 0, they become 0.01 times what they were".)*

Both `forward` and `backward` have to change (the test inside `backward` has to change along
with it). Once you've made the change, run it — can XOR still be solved?

### (b) step: this one is sneakier

```python
def step(value):
    """负数变 0，正数变 1。"""
    return 1.0 if value > 0 else 0.0
```
*(The docstring: "negatives become 0, positives become 1".)*

First think carefully about what its derivative is. (Except at the single point `z = 0`, the
slope is 0 everywhere.)

Now do two experiments:

1. **Change only the function inside `forward`**, leave `backward` untouched down to the last
   character, and run it.
2. Then fix `backward` too (correctly: `step`'s derivative is 0 everywhere, so clip all the
   gradients away), and run it.

The first one still runs, surprisingly. Only the second one actually stops moving.

So now the question: **that first time, the gradient we computed — the gradient of which
function was it?**

## 3. Put relu after the output layer

The order right now is: `middle layer → relu → output layer → softmax`.

Try moving relu to after the output layer: `middle layer → output layer → relu → softmax`.

What happens? Why is this change wrong?

(Hint: the two classes' scores are ultimately there to be **compared**. If both scores turn
into 0, what is there left to compare?)

## 4. Count the parameters, then think it over

The table in `before.py` lists the parameter count for each depth.

A 10-layer network has 60 parameters, a 1-layer one has only 6 — ten times the parameters, and
the loss is exactly the same. Why can't more parameters rescue it?

## 5. Leave the middle layer with only 1 unit, but add relu

Run with `hidden_size` set to 1.

Can the loss come down? If it can't, that shows **width** and **that bend** are two separate
things — the bend is necessary, the width is not.

## 6. Take this bend back to Chapter 6's task

Go back to those 10 sentences from Chapter 6: a vocabulary of 16 words, two classes.

The model right now is `16 numbers → 2 scores`. Give it a middle layer of 8 units, let each
unit pass through relu after it computes, and then output 2 scores.

Derive the gradient for this case by hand (it's the two-layer gradient plus the relu line),
run it, and see whether the loss can get below 0.02.

Once it runs, ask yourself a question: **how many lines of gradient code did I write to add
this one layer? And what if I add another?**
