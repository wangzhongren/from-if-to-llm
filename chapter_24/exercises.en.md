**English** | [中文](exercises.md)

# Chapter 24 Exercises

This chapter has no new mechanism, it's all probes. So the exercises are all "ask the same question a different way".

---

## 1. The grain of the shuffling

`after.py` has:

```python
def shuffle_blocks(text, block_size, seed):
```

The smaller the block, the less local order is destroyed. Write a loop that takes the block size from 1 up to 32:

```python
for block_size in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    result = evaluate_text(model, tokenizer, shuffle_blocks(CORPUS, block_size, SEED))
    print(f"{block_size:>3}  {result['loss']:>9.4f}  {result['accuracy']:>9.2%}")
```

- At what block size does perplexity take off?
- How long a stretch of order can the model roughly "hold on to", according to that threshold?
- Compare it with `CONTEXT = 32`. During training it can see 32 characters at a time — but how much of that does it actually use?

---

## 2. Change the context length — is the conclusion still there

`after.py` has:

```python
CONTEXT = 32
```

Change it to 8 and re-run the whole of `after.py` (it has to retrain, about twenty seconds).

- Did the loss after training change?
- In probe 5's table, is the conclusion "two characters are enough" still there?
- In probe 1's table, is the perplexity on the original corpus still 1.02?
- Now try 64. Why does `build_batches` cut out *fewer* windows once `CONTEXT` gets bigger?

---

## 3. The middle setting of the cross-evaluation

`experiment.py`'s cross-evaluation has only two settings: the original text and the fully shuffled text.

Now add a third: "shuffle every 4 characters".

```python
mild = shuffle_blocks(CORPUS, 4, SEED)
mild_model = train_on_text(tokenizer, mild)
```

Then measure four numbers:

```python
print(evaluate_text(original_model, tokenizer, mild)["loss"])
print(evaluate_text(mild_model, tokenizer, mild)["loss"])
```

- For the model trained on the original text, what is the loss on the "shuffled every 4 characters" text?
- And for the model trained on the "shuffled every 4 characters" text, what is the loss on the original text?
- Why is this setting the hardest to explain? — Hint: part of its local order (within 2 characters) is still correct.

---

## 4. Leave the corpus with just one copy

At the top of `after.py`:

```python
CORPUS = (POEM_A + POEM_B) * 8
```

Change it to `* 1` (only 38 characters) and re-run.

- With only 38 characters in the corpus, how many training windows can `CONTEXT = 32` still cut out?
- What happens to that "how many did it get wrong" number in probe 2?
- Is probe 5's table (context length versus accuracy) still that shape?
- Think: why does "the fewer times the corpus repeats, the more clearly you can see whether it's memorizing" hold?

---

## 5. Give it something it has genuinely never seen

The most uncomfortable thing about this chapter is: our corpus contains no sentence of Chinese that it hasn't seen but that is still grammatical. So the "memorize or learn" judgement can't be made.

Now try to build one. The method: take characters from the two poems and assemble a sentence that **does not appear in the corpus, but still reads more or less smoothly**. For example:

```python
NEW_SENTENCE = "明月光疑是地上霜举头望明月低头思故乡"   # 这是语料里的，不算
```

*(The comment says "this one is in the corpus, so it doesn't count".)*

Try these (judge for yourself which ones "look like Chinese"):

```python
"床前明月低头上霜"
"白毛浮绿水低头思故乡"
"举头望明毛浮绿"
"故乡鹅曲项向天歌水"
```

*(All four are blends stitched from fragments of the two poems: "before my bed, bright moon, lower head, frost on…"; "white feathers float on green water, lower my head and think of home"; "raise my head and gaze at bright feathers floating green"; "homeland, goose, neck curved, toward the sky singing, water". None of them appears in the corpus.)*

For each one, measure `evaluate_text`'s perplexity, then read it once with human eyes.

- Which one has the lowest perplexity? Is it also the one you find smoothest?
- Is there one that reads smoothly to you but has high perplexity? Or the other way round?
- If perplexity and your intuition disagree, which do you trust more? Why?

---

## 6. Answer in one sentence

No code needed.

This chapter laid out two readings:

> **Reading A**: it is just a parrot.
> **Reading B**: it really did learn some structure.

The README says "this chapter does not choose for you". Now it's your turn.

**What is your judgement? Name the one piece of evidence you weight most — the one where, if it didn't hold, your judgement would flip.**

(It's fine if this chapter can't give you an answer. When Chapter 28 talks about scale, this question comes back.)
