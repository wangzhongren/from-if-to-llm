**English** | [中文](README.md)

# Chapter 24: Why It Looks Like It's Thinking

## The problem in this chapter

Last chapter we got two knobs that control how "wild" what it writes is.

But there is one thing we have never done from beginning to end: **examine it.**

The only judgement we've been making is one sentence: "what it writes looks like Chinese."

This chapter adds no new parts, changes no architecture, retrains nothing. It does one thing only: **ask it a few questions, and write the answers down.**

Run `before.py` first. What it does is what we've been doing all along — let it write, and we look.

```text
第一印象：它学会了
  开头：床前
  它写：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿

  开头：鹅鹅鹅
  它写：鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月

  开头：白毛浮绿水
  它写：白毛浮绿水红掌拨清波床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项
```

*(Title: "first impression: it learned it". Three openings — 床前 "before my bed", 鹅鹅鹅 "goose, goose, goose", 白毛浮绿水 "white feathers float on green water" — and what the model writes after each. In all three cases it finishes the poem it was in and then continues straight into the other one: Li Bai's poem and the goose poem, in the right order, with no seam visible.)*

All three passages read smoothly. Right characters, right order — even the rhythm of the poems is right.

Seeing that, your first reaction is bound to be: **it learned Chinese.**

Now try a few openings that aren't in the corpus. Still using those same 33 characters:

```text
  开头：故乡故乡故乡故乡故乡故乡故乡故乡
  它写：故乡故乡故乡故乡故乡故乡故乡故乡鹅曲项项项天歌白毛浮绿白毛浮绿水红掌拨清波床前明月前

  开头：水水水水水水水水水水水水水水水水
  它写：水水水水水水水水水水水水水水水水红红地红白红掌红前前故浮前掌项掌拨前疑地地项项前拨乡

  开头：月月月月月月月月月月月月月月月月
  它写：月月月月月月月月月月月月月月月月光疑地上乡鹅鹅鹅鹅曲曲向曲项向鹅白波举浮绿掌拨掌拨掌
```

*(Three openings made of one character repeated 16 times: 故乡 "homeland", 水 "water", 月 "moon". It echoes the repeat back, then the text drifts into corpus characters in jumbled order — 鹅曲项项项天歌, 红红地红白红掌, 光疑地上乡 — and never recovers.)*

The beginning is still reasonable, the further it goes the messier it gets, and in the end it scatters completely.

Seeing that, your first reaction is also bound to be: **it learned nothing at all, it's just repeating the training data.**

Same weights, same model. Change the opening, and our judgement flips from "it learned it" to "it learned nothing".

The problem is with the method itself — "we read it ourselves". It gives us one sample at a time, and we have no ruler in hand, only a pair of eyes.

## The simplest attempt

Get a different ruler. This ruler doesn't read what the model writes; it asks the model directly: **the next character, which one do you think it is?** Then it compares against the correct answer.

This ruler is called **perplexity**.

```
困惑度 = e 的 loss 次方
```

*(perplexity = e to the power of the loss.)*

You can read it as "**roughly how many candidates are fighting it out in the model's head**":

- perplexity = 1: there is only one candidate in its head, it's certain
- perplexity = 2: it's torn between two candidates
- perplexity = 33 (that is, `e^ln33`): it's guessing blindly among 33 characters — which is exactly our model's vocabulary size

The nice thing about perplexity is that it doesn't need us to "read" the output. It only needs a stretch of **text with a known correct answer**: feed that text to the model, have it guess the next character one position at a time, then average the loss over all the positions.

The more a stretch of text "suits its taste", the lower the perplexity.

So what text does our model have the most appetite for?

## Experiment

`after.py` runs six sets of probes. The first set is the ruler from the last section.

### Probe 1: how certain is it

```text
  这段文字                     loss            困惑度      猜对比例
  -------------------------------------------------------
  语料原文                   0.0162           1.02    99.49%
  每 2 个字打乱一次             9.9164       2.03e+04    47.21%
  每 4 个字打乱一次            16.0393       9.24e+06    21.75%
  每 8 个字打乱一次            19.5929       3.23e+08     8.16%
  整段打乱                  21.4704       2.11e+09     3.92%
```

*(Rows: the original corpus text; the same text with every 2 characters shuffled; every 4; every 8; the whole thing shuffled. Columns: loss, perplexity, fraction guessed right.)*

From the second row on, all we did was "put the same batch of characters in a different order". The set of characters, and how many times each appears, is unchanged.

- Original: perplexity **1.02**. It barely hesitates.
- Fully shuffled: perplexity **2.11e+09** and 3.92% guessed right — while blind guessing is exactly 3.03%.

**Scramble the order and it falls back to blind-guessing level.**

### Probe 2: when it's wrong, does it know

```text
  语料原文一共 8704 个「猜下一个字」的位置。
  它猜错了 44 个（0.51%）。

  它猜对的时候，给正确答案的概率平均是  0.9976
  它猜错的时候，给正确答案的概率平均只有 0.1681
  但它猜错的时候，给**自己选的那个答案**的概率平均有 0.8293
```

*(Over the original corpus there are 8,704 "guess the next character" positions; it got 44 wrong (0.51%). When it is right, the probability it gave the correct answer averages 0.9976. When it is wrong, the probability it gave the correct answer averages only 0.1681 — but the probability it gave **the answer it chose itself** averages 0.8293.)*

The last line is the one to look at: **it is wrong with great confidence.**

It isn't "I wasn't sure, so I picked one at random" — it's "I am very certain, and certain about the wrong thing".

And there are only three kinds of mistake in total:

```text
    正确答案 '光'，它猜 '低'，而且给了自己 98.13% 的把握
    正确答案 '思'，它猜 '望'，而且给了自己 98.99% 的把握
    正确答案 '鹅'，它猜 '曲'，而且给了自己 56.45% 的把握
```

*(The correct answer was 光 "light" and it guessed 低 "low", giving itself 98.13% confidence; the correct answer was 思 "think of" and it guessed 望 "gaze", 98.99%; the correct answer was 鹅 "goose" and it guessed 曲 "curved", 56.45%.)*

These three characters are exactly the **ambiguous** ones in the corpus:

```text
'月' 后面有时是 '光'（床前明月光），有时是 '低'（望明月低头…）
'头' 后面有时是 '望'（举头望明月），有时是 '思'（低头思故乡）
'鹅' 后面有时还是 '鹅'，有时是 '曲'
```

*(After 月 "moon" it is sometimes 光 "light" (床前明月光, "before my bed, bright moonlight") and sometimes 低 "low" (望明月低头…, "gaze at the bright moon, lower [my head]"). After 头 "head" it is sometimes 望 "gaze" (举头望明月, "I raise my head and gaze at the bright moon") and sometimes 思 "think of" (低头思故乡, "I lower my head and think of home"). After 鹅 "goose" it is sometimes another 鹅 and sometimes 曲 "curved".)*

### Probe 3: where its mistakes happen

```text
  窗口里第几个位置        0, 1
  这个位置上错了几个      29, 15

  全部 44 个错误，都发生在窗口的第 0 个和第 1 个位置上。
```

*(Which slot in the window, and how many errors in that slot: slot 0 → 29, slot 1 → 15. All 44 errors happen in the window's 0th and 1st slots.)*

That is: every time the model is forced to decide on one or two characters of context, it makes a mistake. Give it a longer context and it never errs again.

### Probe 4: the characters it's least sure about are the most interesting ones

```text
  它最不确定的 5 个字             它最有把握的 5 个字
  ----------------------------------------------------
  '光'  0.9336                '月'  0.9999
  '鹅'  0.9624                '项'  0.9999
  '思'  0.9694                '上'  0.9998
  '曲'  0.9855                '前'  0.9998
  '低'  0.9959                '天'  0.9997
```

*(Left column: the five characters it is least certain about, with the probability it gives the correct answer — 光 "light" 0.9336, 鹅 "goose" 0.9624, 思 "think of" 0.9694, 曲 "curved" 0.9855, 低 "low" 0.9959. Right column: the five it is most certain about — 月 "moon" 0.9999, 项 "neck" 0.9999, 上 "on" 0.9998, 前 "in front" 0.9998, 天 "sky" 0.9997.)*

The least certain ones, 光 鹅 思 曲 低, all sit in ambiguous positions.

The most certain ones, 月 项 上 前 天, have only one possible continuation in the corpus, no alternative at all.

**Its uncertainty is not random.** It is uncertain in exactly the places where it ought to be uncertain.

### Probe 5: how much context does it need before it dares to be certain

```text
  给它的上下文长度                   下一个字的损失      猜对比例
  ----------------------------------------------
  1                           0.2336    88.64%
  2                           0.1739    95.35%
  3                           0.0006   100.00%
  4                           0.0008   100.00%
  8                           0.0017   100.00%
  16                          0.0093   100.00%
  32                          0.0007   100.00%
```

*(Context length given to it, against the loss on the next character and the fraction guessed right.)*

Give it two characters and it already gets the next character right 95% of the time. Give it 32 and it doesn't do any better.

**This model doesn't really need much context.** Because our corpus is so regular: what follows a character is almost always determined by the one or two characters before it.

### Probe 6: can it recite a whole phrase

```text
    '床前明月光疑是地上|' -> '霜'   （语料里是 '霜'）
    '举头望明月低头思|' -> '故'   （语料里是 '故'）
    '白毛浮绿水红掌|' -> '拨'   （语料里是 '拨'）
```

*(The `|` marks where the prompt stops. From the first prompt it answers 霜 "frost", from the second 故 (as in 故乡, "homeland"), from the third 拨 "paddle" — and the parentheses say the corpus has exactly that character in each of those places.)*

All three correct. But that doesn't show much — the model **saw each of these fragments 8 times** during training.

Is it "continuing the poem" or "recalling it"? This chapter cannot answer that. Because in our corpus there has never once been a sentence of Chinese that it hasn't seen but that is still grammatical.

### So, one more control experiment

`experiment.py` has a harsher one:

```text
  原文：  床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿
  打乱后：掌望项绿明霜光前清明床前思床头鹅明清天地地头浮鹅拨是歌绿地月月月

                                考它原文          考它打乱文本
  --------------------------------------------------
  用原文训练的模型                  0.0162         21.4704
  用打乱文本训练的模型             14.0036          0.1286
```

*(Top line: the original text — the first 40 characters of the corpus, "moonlight before my bed…". Second line: the same 40 characters shuffled, "same characters, different order". The table then tests each of two trained models on each of the two texts: the model trained on the original text scores 0.0162 on the original and 21.4704 on the shuffled text; the model trained on the shuffled text scores 14.0036 on the original and 0.1286 on the shuffled text.)*

The two texts use **exactly the same characters**, just in a different order. The model structure, the number of training steps, the random seed — all identical.

**The two numbers on the diagonal are small; the other two are large.**

Each model only recognizes the ordering it saw during training. Move it onto the other ordering and it instantly reverts to blind guessing.

This diagonal is the hardest piece of evidence in the chapter:

> **What it learned is "what order these 304 characters go in", not "what order Chinese characters go in".**

Because in the shuffled text, not one fragment follows the habits of Chinese — and yet it learned that text just as well as the original. It just learned the rules of that scrambled ordering.

Now train it for longer:

```text
      训练步数       原文上的 loss         打乱文本上的 loss
  ----------------------------------------------
       200          0.0180             17.6676
       500          0.0111             19.1756
      1000          0.0162             21.4704
      2000          0.0167             23.9451
```

*(Training steps against loss on the original text and loss on the shuffled text.)*

The loss on the original text sits between 0.01 and 0.02 the whole time; it hit bottom long ago. The loss on the shuffled text doesn't just fail to fall — it climbs from 17.7 all the way up to 23.9.

(The blind-guessing level is `ln33 = 3.50`. These are far above it, which means the model is "confidently wrong".)

**The longer it trains, the better it recites this passage, but it does not understand Chinese one bit better. It even gets worse at coping with an unfamiliar ordering the longer it trains.**

## The new mechanism

**This chapter has no new mechanism.**

No new layer, no new operator, not one line of model code changed. Only two things were added:

1. **A ruler**: perplexity. It doesn't need us to "read" the output, it only needs text with known answers.
2. **A control**: shuffle the corpus, train a second model, and see whether it can survive on the other one's turf.

That's the whole chapter. It produces no new capability; it only **tells us what the thing we built actually is**.

## Python implementation

The ruler itself is a few lines:

```python
def perplexity(loss):
    return float(np.exp(loss))


def evaluate_text(model, tokenizer, text):
    ids = tokenizer.encode(text)
    inputs = [ids[s:s + CONTEXT] for s in range(len(ids) - CONTEXT)]
    targets = [ids[s + 1:s + CONTEXT + 1] for s in range(len(ids) - CONTEXT)]
    with no_grad():
        logits = model(np.array(inputs)).data
        loss = cross_entropy(model(np.array(inputs)), np.array(targets)).item()
        correct = logits.argmax(axis=-1) == np.array(targets)
    return {"loss": loss, "perplexity": perplexity(loss),
            "accuracy": float(correct.mean()), ...}
```

The shuffler is a few lines too:

```python
def shuffle_whole(text, seed):
    characters = list(text)
    np.random.default_rng(seed).shuffle(characters)
    return "".join(characters)
```

`shuffle_blocks` is its gentler version: it only shuffles within each small block. The bigger the block, the more thoroughly it destroys the text.

These two functions together are under ten lines. But they let us run a **control experiment** on the model for the first time.

## What it solves

This chapter did not make the model stronger. It did something else: **it stopped us from fooling ourselves.**

Here is what we know now:

| What we saw | The number |
|---|---|
| Perplexity on the original corpus | 1.02 (only one candidate in its head) |
| Perplexity after shuffling the whole text | 2.11e+09 (back to blind guessing) |
| Total errors | 44 wrong out of 8,704 positions |
| Kinds of error | only three, all at ambiguous positions in the corpus |
| Confidence when it's wrong | 82.93% on average |
| Where the errors happen | always slots 0 and 1 of the window |
| The diagonal of the cross-evaluation | 0.0162 / 0.1286 against 21.47 / 14.00 |

And it overturns a conclusion we would otherwise have reached.

If we only ran the shuffling experiment, the conclusion would be "it's just a parrot". But probes four and three say something else:

**A pure lookup table would not be "especially hesitant at a particular position".**

The five characters it is least certain about are exactly the five that are genuinely ambiguous in the corpus. It makes mistakes with two characters of context, and no mistakes with three. A model that simply memorized 304 characters would not need to do any of that.

So:

- **Reading A**: it is just a parrot. It knows nothing outside the corpus, and it can't even recognize the corpus once the order is scrambled.
- **Reading B**: it really did learn some structure from those 304 characters — it knows which positions are ambiguous, where it should hesitate, and whether two characters are enough for it to decide.

**This chapter does not choose for you.** Both sets of observations hold at the same time, and both make sense.

## What it still can't solve

One thing is certain: **the questions we're asking can't be answered at this scale.**

304 characters, two poems eight times over. 150,000 parameters against 304 characters is a ratio of 500 to 1. Of course it can memorize that. To force it to learn structure, we'd have to make it unable to memorize — **the corpus has to be too big to remember.**

So make it bigger?

For it to genuinely learn Chinese, the parameter count would have to reach the hundreds of millions, and the corpus would have to reach billions of characters.

Neither of those is something "changing a few constants" can fix. But before we even start, we hit a different wall:

**The `toygrad` we wrote in Chapter 9 is too slow.**

Every operation has to be recorded by Python, and every step has to walk the whole computation graph from the start. Our 150,000-parameter little model takes over ten milliseconds per step. What about 1,000 times bigger?

**Actually, there's an even more basic problem we haven't solved.**

Everything we measured in this chapter rested on **every line of code being handwritten by us**: hand-written attention, hand-written LayerNorm gradients, hand-written Adam, hand-written training loop. All of this exists ready-made in PyTorch, and much faster.

Our model can't grow right now. The first reason isn't that we lack a GPU — it's that **the tooling we built can't keep up**.

Next chapter we switch tools. But remember one thing first: **switching tools will not change a single principle.** Every line you wrote today has an identical counterpart in PyTorch.

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Raise `shuffle_blocks`' block size from 1 to 32 and draw a perplexity curve. At what block size does perplexity take off? What does that threshold say about how long a stretch of order the model can roughly "hold on to"?
2. Probe 5 only measures up to 32 characters of context. Change `CONTEXT` to 8, 16, 64 and retrain plus re-probe; is the conclusion "two characters are enough" still there?
3. In `experiment.py`'s cross-evaluation, if the shuffling were changed to "shuffle every 4 characters", what would the four numbers become? Think about why that middle setting is the hardest to explain.
4. In probe 5's table, the accuracy at context length 1 is 88.64%. Change the corpus to `(POEM_A + POEM_B) * 1` (only repeated once) and train again — does this number go up or down? Why? (Hint: the fewer times the corpus repeats, the harder it is for "memorizing" to hold up.)
