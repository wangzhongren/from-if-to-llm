**English** | [中文](exercises.md)

# Chapter 25 Exercises

## 1. Swap the hand-written attention for the packaged one

`Block.attention` in `after.py` is ours (split heads, score, mask, softmax, stitch back).

Replace it with `nn.MultiheadAttention`:

```python
self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
...
out = self.attn(x, x, x, attn_mask=mask, need_weights=False)[0]
```

Then run `after.py` and compare against the original loss curve.

Hint: experiment 3 in `experiment.py` already works out how to move the weights across.
You can copy the way it assembles `in_proj_weight` to load our weights into
`nn.MultiheadAttention`, and that lets you check whether the training curve is numerically
the same after the swap.

## 2. Use the Transformer layer that ships with PyTorch

PyTorch has a ready-made `nn.TransformerEncoderLayer`. Replace our `Block` with one layer
of it:

```python
layer = nn.TransformerEncoderLayer(dim, n_head, dim_feedforward=4*dim,
                                   batch_first=True, activation="relu")
x = layer(x, src_mask=mask)
```

It has two arguments worth fiddling with:

- `norm_first`: `False` is "compute first, LN after" (our way), `True` is "LN first, then
  compute" (GPT's way — Chapter 27 explains why).
- `activation`: `"relu"` or `"gelu"`.

Run all four combinations and see whether the loss differs on our small model.
(Spoiler: barely. Chapter 27 explains why "barely" is itself the point.)

## 3. Make the model bigger and watch the multiplier

Open `experiment.py` and change `BENCH_SIZES`:

```python
BENCH_SIZES = {
    "80 万参数": (dict(dim=128, n_layer=4, n_head=4, block_size=32), 20),
    "800 万参数": (dict(dim=256, n_layer=8, n_head=4, block_size=32), 5),
}
```

(the keys read "800 thousand parameters" and "8 million parameters")

Run it again and look at the "multiplier" column. Two things to notice:

- How much did memory grow on our engine's two rows? Can your machine still hold it?
- Is the speed multiplier getting bigger or smaller? Why?

## 4. Verify the transpose by hand

In `load_our_weights`, every weight gets a `.T`. Remove one of the `.T`s, make the program
either crash or produce something completely unreliable, and then explain clearly:

- What shape is `nn.Linear`'s `weight`?
- Does its `forward` actually compute `x @ W` or `x @ W.T`?
- Without the transpose, can the model still train? What does the loss look like?

## 5. Try turning off `no_grad`

`after.py`'s `generate` has a `@torch.no_grad()` line above it. Delete it and run again.

- Are the results the same?
- What happens to memory and time?
- Why don't we need a computation graph when generating? (Recall Chapter 23: what are we
  doing when we generate, and is there an "answer" to compare against?)

## 6. Make `before.py` crash once

The big model in `before.py` only ran 5 steps. Change the step count to 50 and watch on
your own machine how far memory climbs and when it starts to crawl.

That is half the answer to the Chapter 24 question: **it isn't only that the model is too
small — our tools are too small too.**
