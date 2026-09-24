**English** | [中文](exercises.md)

# Chapter 28 Exercises

## 1. Structure vs parameters

`SIZES` in `after.py` currently changes "layer count and dimension" together. Now change
only one of them:

```python
SIZES = [
    ("小", dict(dim=32, n_layer=1, n_head=4)),
    ("中", dict(dim=64, n_layer=2, n_head=4)),
    ("大", dict(dim=128, n_layer=5, n_head=4)),
]
```

(the names read "small", "medium", "large")

Try these variations, running each one, and fill the results into the table:

| Configuration | Parameter count | Validation loss at 150 steps |
|---|---|---|
| Original (1 layer / 2 layers / 5 layers) | | |
| All changed to 1 head (`n_head=1`) | | |
| All changed to 8 heads (`n_head=8`) | | |
| All at 1 layer with a wider dimension (try `dim=128, n_layer=1`) | | |

- Does piling parameters into "deeper" and piling them into "wider" work the same way?
- With the head count changed but the parameter count unchanged, how much does the loss
  move?

## 2. The data term

Cut the corpus from 12,000 sentences to 1,500 (change only `build_corpus`'s default
argument), change nothing else, and re-run `experiment.py`.

- What happens in the 600-step column? Does the ordering of the three models change?
- On which model is the gap between training loss and validation loss (that is, how much
  it memorized) the largest?
- Which letter of the scaling law (N / D / C) does this experiment correspond to?

## 3. Treat the learning rate as a knob too

Right now all three models share the same learning rate of 0.003.
Tune the learning rate separately for each model (try 1e-3, 3e-3, 1e-2) to make its
validation loss at 150 steps as low as possible.

| Model | Best learning rate | Its validation loss |
|---|---|---|
| Small | | |
| Medium | | |
| Large | | |

- Does the big model need the same learning rate as the medium one? (Recall the gradient
  experiment in Chapter 27.)
- If every model is given its best learning rate, does "bigger is better" still hold? Does
  the gap get smaller or bigger?

## 4. Draw a scaling-law chart by hand

Plot the three models' (parameter count, best validation loss) with a logarithmic
horizontal axis:

```python
import math
for row in results:
    x = math.log(row["params"])          # take the log of the parameter count
    y = math.log(row["loss"])            # and the log of the loss
    print(f"{x:.2f}  {y:.2f}")
```

- Do the points lie roughly on a straight line? (A power law is a straight line in
  log-log coordinates.)
- Three points determine a line. Extrapolate along it: how many parameters would you need
  to bring loss down to 0.45? Is that extrapolation trustworthy? Why or why not? (Hint:
  think about the "loss floor" from this chapter.)

## 5. Make the statistical model stronger

The n-gram in `experiment.py` uses "backoff plus smoothing", which is very plain. Try:

- Changing `ALPHA` from 0.5 to 0.01 and to 5.0 and watching how the loss moves.
- Changing `max_order` from 8 to 12 (watch out for memory and time). Can the loss get
  below 0.7?

Then answer: **why can't an n-gram catch up with our model no matter how many orders you
add?** (Hint: with 8 characters of context, how many times does each context occur on
average in a 160,000-character corpus?)

## 6. Use loss to decide whether to keep training

Take the trained "large" model from `after.py` and train it another 600 steps (1200 in
total), recording training loss and validation loss every 100 steps.

- At what point do the two curves diverge?
- After they diverge, what happens to validation loss if you keep training?
- If you were the engineer, what would you decide when you saw that divergence? (Hint: add
  data? shrink the model? stop?)
