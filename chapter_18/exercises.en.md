**English** | [中文](exercises.md)

# Chapter 18 exercises

## 1. Find "from which layer it runs away"

Set `before.py`'s `N_LAYERS` to 2, 4, 8, 16 in turn, run each one, and note down the standard deviation "received by layer 8" (if there aren't that many layers, look at the last one).

**Question**: is this number linear in the layer count (`L`), square-root (`sqrt(L)`), or exponential? Use the numbers you wrote down to decide. One more: compute the "hands on / receives" ratio for each layer — is it a constant?

## 2. What are the weight / bias after the normalization for

Change every `ln%d_w` initialization in `after.py` to 3.0 (`np.full(d_model, 3.0)`) and run the statistics table and the training again.

**Questions**:
- Is the "receives" column still 1.000?
- What does the "hands on" column become?
- Does it still learn? (This one is meant to show: normalization is not "make the numbers smaller", it is "make the scale controllable".)

## 3. Move the normalization to after the residual

Take these two lines in `after.py`:

```python
handed = layer_norm(x, self.p["ln%d_w" % i], self.p["ln%d_b" % i])
x = x + self.attention(handed, layer)
```

and change them to:

```python
x = layer_norm(x + self.attention(x, layer), self.p["ln%d_w" % i], self.p["ln%d_b" % i])
```

(Note that now attention receives an **un-normalized** x.)

**Question**: does 8 layers still learn? Roughly what loss does it reach? Then think again: after this change, is the residual's through lane still there? (Hint: for a gradient to get back to the previous layer, it has to pass through the normalization first.)

## 4. Implement an RMSNorm yourself

"The mean step can be dropped" — divide by the root mean square only:

```text
rms = sqrt(mean(x 的平方))
y = x / rms * weight
```

*(`x 的平方` = "x squared": the root mean square of x, then scale by `weight`.)*

In `after.py`, replace `layer_norm` with this version you wrote yourself (note: no `bias` is needed here) and run 8 layers.

**Question**: how does the loss compare with LayerNorm? Why does "not subtracting the mean" still work? (Hint: what gets subtracted is a constant, and this step happens before a linear transform — the linear transform can put that constant back itself, as long as there is a bias after it.)

## 5. Think about it: why "the dumbest approach" is fine for images

"Put the whole batch together and normalize it" is actually very common in image models (the kind from Chapter 8), and it works fine there.

**Question**: when an image model trains, how many images are in a batch? And at inference time? Why isn't it afraid of the "a batch holds only one sample" problem? (Hint: think about whether the batch size is the same at training and at inference; then think about where the "whole-batch statistics" get stored.)

## 6. Measure: after normalization, is the learning rate still easy to tune

In `after.py`, change the learning rate from 0.05 to 0.2 and 0.5, running 100 steps each; then do the same in `before.py` (with the layer count set to 4).

**Question**: which side is more sensitive to the learning rate? What does that have to do with Chapter 6's "if the learning rate is too large, you overshoot"?
