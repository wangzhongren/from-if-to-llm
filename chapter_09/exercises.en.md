**English** | [中文](exercises.md)

# Chapter 9 Exercises

## 1. Add an operator to the engine

The `Value` in `after.py` only has `+ - * / ** exp log relu`.

Give it a `tanh`:

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward():
        self.grad += (1 - t * t) * out.grad
    out._backward = _backward
    return out
```

Once you've added it, check it with the numeric method (copy how
`test_scalar_engine_matches_numeric_gradient` is written).

Then try adding a `maximum(other)` yourself (take the larger value; `maximum(x, 0)` is relu).
Hint: when `x` is larger than `other`, the gradient goes to x; otherwise it goes to other.

## 2. Change `+=` to `=`

Change every `self.grad += ...` in `_backward` to `self.grad = ...`.

Then run this:

```python
x = Value(3.0)
(x * x).backward()
print(x.grad)
```

What does it come out as? What should it be? Where's the difference?

Change it back, then run a full training round again and see whether the loss still goes down
properly.

## 3. Delete the topological sort

The current `backward()` sorts first, then walks in reverse.

Change it to skip the sort and just recurse downwards:

```python
def backward_broken(node):
    node.grad = 1.0

    def go(n):
        n._backward()
        for child in n._prev:
            go(child)

    go(node)
```

Then run this piece:

```python
a = Value(2.0)
b = a * 3.0     # b = 3a
c = b * a       # c = 3a²
d = b + c       # d = 3a + 3a²
backward_broken(d)
print(a.grad)
```

The correct answer is `3 + 6a = 15`. What does it actually come out as?

Where does the extra part come from?

(Hint: `b` has two downstream nodes — `d` and `c`.
Without the sort, `b._backward()` may run before `c` has given its gradient to `b`.
At that point `b.grad` hasn't been fully collected, and it passes on a value that is too small.
Count it out: how much too small was it? How much extra did it end up computing?)

## 4. Why backward sets it to 1.0 at the start

The first statement of `backward()` is `self.grad = 1.0`, not `self.grad += 1.0`.

If you call `loss.backward()` twice, is loss's own gradient 1 or 2?

Does this matter? Think about it together with "why parameters have to be cleared".

## 5. Wrap `forward` in `no_grad()` during training

Build a small network with `toygrad`, then write this in the training loop:

```python
opt.zero_grad()
with no_grad():
    logits = (Tensor(X) @ w1 + b1).relu() @ w2 + b2
    loss = cross_entropy(logits, Y)
loss.backward()
opt.step()
```

Run 50 steps and print the loss at each one.

What happens? Did the program raise an error? Did the parameters move?

(This exercise is worth doing for exactly this: **all the gradients become 0, and there is not
a single error.
Training just quietly sits there.** Later on, when you write one wrong line in PyTorch and see
the loss refuse to move, think of this first.)

## 6. Computing 4 samples at once vs one at a time

In `after.py` we add the four samples' losses together and use that as the total loss.

Now change it to **one at a time**: call `backward()` after each sample is computed, let the
four samples' gradients pile up, and then update the parameters once.

Run it, and compare with the original: does the loss fall at the same rate? Is the final
accuracy the same?

Think it through carefully: are the gradients the two approaches compute the same thing?
(Hint: the derivative of a sum = the sum of the derivatives. But the **accumulate** property
is what saves us here — if `backward()` overwrote instead of accumulating, the second way of
writing it would be dead on arrival.)

## 7. Work through a graph by hand

Take `a = 2`, `b = 3`, `c = 4` and draw the computation graph of `loss = a * b + c`.

Then walk through it on paper:

- what each node's `data` is;
- what each node's `grad` is;
- which number travels along each edge.

When you're done, change `a` to `2.001`, recompute `loss`, and see how much it changed.
Does that match the `a.grad` you worked out on paper?

## 8. Where this engine is slow

`after.py` runs 300 steps (4 samples per step) in well under a second.
But the array version in `toygrad` does the same task in a few tens of milliseconds.

Count it out: how many `Value` objects does the scalar version create per sample?
If there are 1 million training examples, what does that number become?

(This is why real frameworks use arrays instead of computing one number at a time.
But the idea behind both is exactly the same as the `Value` you wrote by hand.)
