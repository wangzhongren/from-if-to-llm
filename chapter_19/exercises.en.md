**English** | [中文](exercises.md)

# Chapter 19 exercises

## 1. Swap in a linear label

Take this line of `make_probe` in `after.py`:

```python
y = ((x[:, :, 0] > 0) ^ (x[:, :, 1] > 0)).astype(int)
```

and change it to "or" (1 if either of the two numbers is positive):

```python
y = ((x[:, :, 0] > 0) | (x[:, :, 1] > 0)).astype(int)
```

Run each of the three structures.

**Question**: "or" is linear (one straight line separates it), "xor" is not. This time, can the Attention-only one learn it? Why? What does the contrast show?

## 2. Can the middle layer skip the widening

In `after.py`, change `W1`'s shape from `(32, 128)` to `(32, 32)` and `W2` from `(128, 32)` to `(32, 32)` (changing every related place too), then run 600 steps.

**Question**: how far does the accuracy fall? If it falls, does that mean "processing" needs "a middle that is wide enough", or just "a ReLU somewhere"? Think about it: if the middle isn't widened, how much more does `linear → ReLU → linear` actually have than a single linear layer?

## 3. Take the ReLU out

Remove the `.relu()` from `mlp`:

```python
def mlp(self, x, block):
    return (x @ block["W1"] + block["b1"]) @ block["W2"] + block["b2"]
```

Run 600 steps.

**Question**: what is the accuracy? In "two linear layers with a ReLU in between", which part is the one actually doing the work? (Hint: two linear layers stacked on top of each other are equivalent to what?)

## 4. Per-layer statistics for the complete block

Following the approach in Chapter 18's `after.py`, print the standard deviation of "what each layer receives" for this chapter's block too.

**Question**: a block contains two normalizations (`ln1` for Attention, `ln2` for the MLP). Which tensor does each of them hold down? If we keep only `ln1` and remove `ln2`, is training still stable?

## 5. Can position 3 see position 5

Write a small experiment in `after.py`: add 100 to everything at position 5 of the input (`x[:, 5] += 100`), then look at how much position 3's output changed.

**Questions**:
- In the MLP-only model, how much did position 3's output change?
- And in the Attention-only model?
- And in the model with both?
- Then one level deeper: if position 5 is shifted **as a whole** by a constant (100 added to every dimension), why might the Attention side see no change at all after normalization?

## 6. Think about it: can Attention and the MLP substitute for each other

This chapter's experiment says: only Attention can exchange, only the MLP can process.

**Question**: so can "two MLPs stacked" replace one Attention? Can "two Attentions stacked" replace one MLP? (Hint: think from these two angles — can one position see another position, and can the output leave the input's range of values.)
