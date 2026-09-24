**English** | [中文](exercises.md)

# Chapter 27 Exercises

## 1. How many layers does it take before things go wrong

In `experiment.py`, change the layer count from 12 to 4, 8, 16, and 24, running each one,
and measure both post-LN and our ordering:

| Layers | Our ordering (bottom-layer gradient at step 30) | post-LN |
|---|---|---|
| 4 | | |
| 8 | | |
| 12 | | |
| 24 | | |

Guess first, then run. Then answer:

- With few layers, why do both orderings learn?
- Is "post-LN collapses at 12 layers" a matter of depth or of learning rate? Try dropping
  the learning rate from 0.01 to 0.003 and see.

## 2. Rescue it with a learning-rate warmup

Give the post-LN model a learning-rate warmup:

```python
def lr_at(step, warmup=50, base=1e-2):
    if step < warmup:
        return base * (step + 1) / warmup
    return base
```

(in the training loop, set `optimizer.param_groups[0]["lr"] = lr_at(step)`.)

- Can it be rescued?
- If so, what does that say about the relationship between "the learning rate is too big"
  and "the normalization is in the wrong place"?
- Warmup is standard practice in GPT training. Now you know why it is needed.

## 3. What exactly is that LayerNorm before the output layer doing

In `after.py`, `final_ln=True` is GPT's choice. Now run a set of comparisons:

| Layers | Validation loss with `final_ln=True` | Validation loss with `final_ln=False` |
|---|---|---|
| 4 | | |
| 12 | | |
| 24 | | |

- Does the difference get more obvious the more layers there are?
- Think about why: the pre-LN residual stream "just keeps adding up", so as the layer
  count grows, what happens to the scale of that final vector? What's the problem with
  feeding it straight into a classifier?
- Verify your guess in code: print the standard deviation after each layer's output
  (`hidden.std()`) and see whether it grows or shrinks with depth.

## 4. Swap the position table back to the sinusoidal one

In `after.py`, `learned_pos=True` uses a learnable position table. Set it to `False`,
retrain, and compare:

| | Validation loss | Generated sentences |
|---|---|---|
| Learnable position table | | |
| Fixed sinusoidal table | | |

Then think about this: if at test time you hand it a sentence longer than `block_size`,
what happens to each of the two approaches? (Hint: a table lookup goes out of bounds; a
sinusoidal table can compute longer positions.)

## 5. Verify pre-LN's two properties by hand

The test `test_我们的模型和第_22_章一样是_pre_ln` in the test file verifies that
"scaling the input up 10 times leaves the sublayer's output unchanged". Now verify the
second property:

```python
with torch.no_grad():
    hidden = model.tok_emb(x) + positions
    for i, block in enumerate(model.blocks):
        hidden = block(hidden, mask)
        print(i, hidden.std().item())
```

Run it once for pre-LN and once for post-LN:

- How do the two curves differ?
- Why doesn't post-LN need that final LayerNorm, while pre-LN does?

## 6. Make the 12-layer model "think" a little

Take the trained 12-layer model and try a sentence it shouldn't know:

```python
print(generate(model, "苹果", n_new=40, temperature=0.5, seed=3))
print(generate(model, "香蕉发布新手机", n_new=40, temperature=0.5, seed=3))
```

The second one is a combination that can never occur in the corpus (bananas don't ship
phones). Watch how it continues:

- Does it treat that sentence as a tech topic or a food topic?
- From which characters onward does it "decide" the topic?
- Does this count as "understanding"? (Chapters 23 and 24 asked this once; now you have a
  bigger model to ask it a second time.)
