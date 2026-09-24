**English** | [中文](exercises.md)

# Chapter 17 exercises

## 1. See the 8-layer nan for yourself

In `after.py`, change `AttentionStack(n_layers=4)` at the bottom to `n_layers=8` and run it.

You will see the loss start in the tens or hundreds at step 1, and then turn into `nan` very quickly.

**Question**: at which step does `nan` first appear? Change the print interval to every 10 steps and find out.

## 2. How many layers does the old version really survive

In `before.py`, set `n_layers` to 1, 2, 3, 4 in turn, run each one, and fill the final loss into a table.

**Question**: does it degrade gradually, or fall off a cliff at some particular layer? Does that "breaking point" have anything to do with the length of the two sentences (11 words)?

## 3. Discount the residual

Take this line in `after.py`:

```python
x = x + self.attention(x, layer)
```

and change it to:

```python
x = 0.5 * x + self.attention(x, layer)
```

Then run 4 layers and 8 layers again.

**Question**: can 8 layers run now? Why is discounting the residual not the same as "adding fewer times", but rather "every time you add, the old information decays a little"? (Hint: expand 3 layers and look at what each term inside `x` is multiplied by.)

## 4. Measure "the numbers are growing"

Add a line of statistics inside `after.py`'s `forward`:

```python
for layer in self.layers:
    x = x + self.attention(x, layer)
    print(f"标准差 = {x.data.std():.4f}")
```

(The printed label `标准差` means "standard deviation".) Run it with 1 layer, 2 layers and 4 layers.

**Question**: after 4 layers, how many times larger is the standard deviation than at layer 1? If the layer count is L, what do you think it roughly scales with (`L`? `sqrt(L)`? `2**L`)? Does it still hold after switching to `x = 0.5 * x + ...`?

## 5. Can a residual rescue a "shallow" model

Set `before.py` back to 1 layer, then change the line in its forward to `x = x + self.attention(x, layer)` and run it.

**Question**: compared to the original 1-layer version, is it better, worse, or about the same? Does that mean the residual connection solves the "depth" problem, or the "fitting capacity" problem?

## 6. Think about it: why "copying" matters so much

The laziest solution to this chapter's task is "copy the input". Let's rephrase that:

> An `L`-layer network, if it wants to express "do nothing", how much does it have to learn?

**Question**: in the "replacement" style, every layer has to learn an identity map; in the "add" style, every layer only has to learn an output close to 0. Which is easier to learn? Think back to ReLU in Chapter 8: to make `relu(x)` output exactly 0, you only need the weights to be negative — whereas to make a whole attention layer approximately the identity, how well would `Q`, `K`, `V` and `O` all have to cooperate?
