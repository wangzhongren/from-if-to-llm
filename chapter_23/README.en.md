**English** | [中文](README.md)

# Chapter 23: Why a Model "Generates"

## The problem in this chapter

Last chapter ended with the model writing a poem:

```text
  床前明月光疑是地上霜举头望明月低头思故乡
  鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前
  明月光疑是地上霜举头望明月低头思故乡鹅鹅
```

*(The continuation from last chapter, wrapped by us: Li Bai's *Quiet Night Thoughts* in full, then Luo Binwang's *Ode to the Goose*, then Li Bai's poem again — 60 characters of connected Chinese.)*

But that poem was stitched together by our hand-rolled loop. And there's something hidden in that loop that we never mentioned:

**At every step, we took the character with the highest probability and threw away the other 32 probabilities.**

Run `before.py` first, and see what always throwing them away does.

Problem one: the same opening gives the same output no matter how many times you run it.

```text
  开头：举头望明月低头
    第 1 次：举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清
    第 2 次：举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清
    第 3 次：举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清
```

*(Opening: "I raise my head and gaze at the bright moon, I lower my head". All three runs write the exact same 30 characters: 思故乡, "think of home", finishing the poem's line, then the goose poem's opening — 鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清, "goose, goose, goose, neck curved, singing to the sky, white feathers float on green water, red feet pad".)*

Three runs, completely identical. Not a coincidence — that `greedy_continue` in `before.py` contains no random number at all. Given the same opening, the output is **computed**.

Problem two: the moment the opening leaves the corpus, it falls apart quickly.

```text
    故乡故乡故乡故乡...
      -> 故乡故乡故乡故乡故乡故乡故乡故乡鹅鹅鹅曲疑项鹅拨低鹅绿鹅歌曲歌水绿浮明光清拨床乡乡毛

    月月月月月月月月...
      -> 月月月月月月月月月月月月月月月月光光项地是是头天上霜床前是头举清乡地向地上举月歌疑乡
```

*(Two openings that never occur in the corpus: repeated 故乡 "homeland" and repeated 月 "moon". The model echoes the repeated character back for a while, then wanders off into corpus characters in scrambled order — 鹅鹅鹅曲疑项鹅拨低鹅绿鹅歌曲歌水绿浮明光清拨床乡乡毛 and the like.)*

The first few characters are still reasonable; the further it goes the more it scatters. Greedy looks only at "what looks most likely right now" at each step, and one wrong step can't be taken back. It has no option to "step back and look again".

Problem three: what it writes is always the same 38-character cycle.

```text
    床前明月光疑是地上霜举头望明月低头思故乡
    鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前
    明月光疑是地上霜举头望明月低头思故乡鹅鹅
    鹅曲项向天歌白毛浮绿水红掌拨清波床前明月
    光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲
    项向天歌白毛浮绿水红掌拨清波床前明月光疑
```

*(Six lines of 20 characters each, and all six are nothing but the two poems cycling: Li Bai's poem, the goose poem, Li Bai's poem again, and on. The line breaks are ours, but the loop is the model's.)*

This one can't all be blamed on greedy — the corpus is only 38 characters in total. But the first two problems are its own.

**At each position the model gives you 33 probabilities, not 1 character.** What we just did was: take the largest one and throw away the other 32.

What was in those 32 that we threw away?

## The simplest attempt

First, give that loop a name.

What we have been doing all along is this:

1. Feed the characters written so far to the model
2. Get the probability distribution over the next character
3. Pick one and append it
4. Go back to step 1

"Feed the output back in as input, then predict the next one" — that move is called **autoregressive**. We used it all through Chapter 22, we just hadn't named it yet.

And step 3 — "pick one" — is this chapter's real subject.

The most intuitive approach is **greedy**: take the largest one every time. That's the version from Chapter 22. `before.py` has already shown it isn't good enough.

So the remaining choice is: **draw once from those 33 probabilities.**

## Experiment

Before we draw, install two knobs.

**Knob one: temperature.**

```python
adjusted = probabilities ** (1.0 / temperature)
adjusted = adjusted / adjusted.sum()
```

What temperature does is **redistribute the probabilities**:

- temperature = 1: nothing changes
- temperature < 1: the big get bigger and the small get smaller (the model becomes conservative)
- temperature > 1: everyone gets flattened out (the model becomes adventurous)

Same opening, only the temperature changes:

```text
    开头：举头望明月低头思

    温度  0.2  -> 故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑
    温度  0.5  -> 故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑
    温度  1.0  -> 故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑
    温度  2.0  -> 故头思故乡鹅乡鹅曲项向天歌白毛浮绿水红项向天项向天红
    温度  3.0  -> 故望明月低头乡鹅鹅鹅歌天歌毛毛毛鹅拨清鹅拨向拨思头是
    温度  4.0  -> 故望明月低头上霜霜故故乡望明月头霜故故鹅举向天低向是
    温度  6.0  -> 故浮望明月月乡鹅鹅明故低头故前明鹅曲项白向向故乡举头
```

*(Opening: "I raise my head and gaze at the bright moon, I think of". The 0.2, 0.5 and 1.0 rows are character-for-character identical: after 思 it writes 故乡, finishing the line 低头思故乡, then the goose poem in full, then starts Li Bai's poem again at 床前明月光疑. Only from 2.0 does it start to wobble — the words begin to come out in the wrong order; at 4.0 and 6.0 the text is corpus characters shuffled loose.)*

Wait. The first three temperatures (0.2, 0.5, 1.0) write **exactly the same thing**.

Temperature 0.2 and temperature 1.0 differ by a factor of five — how can there be no difference?

`experiment.py` measures it:

```text
     温度       有效候选数       6 次里有几种      最长复读
  --------------------------------------------
    0.2        1.00             1        32
    0.5        1.00             1        32
    1.0        1.00             2        32
    1.5        1.02             3        32
    2.0        1.11             6        32
    3.0        1.62             6        24
    4.0        2.77             6        12
    6.0        6.98             6        11
    8.0       12.09             6         9
```

*(Columns: temperature; effective number of candidates; how many distinct continuations showed up in 6 runs; longest repeat. "Longest repeat" is the longest stretch of the output that matches the corpus.)*

The "effective number of candidates" is what `exp(entropy)` comes out to after temperature is applied to the probabilities. 1 means it is really only seriously considering one candidate; 33 means it is guessing blindly among all 33.

For this opening, the model's highest probability is **0.9999**, and the second highest is **0.000086**.

So after temperature 0.5 the distribution is `0.9999² : 0.000086²`, which is `0.9998 : 0.0000000074`. The runner-up still doesn't get drawn.

For it to be drawn at all, temperature has to be turned past 2. And by then what it writes no longer looks much like language (the longest repeat drops to 12).

**This isn't the temperature knob being broken — it's the model being too confident.** The distribution it hands out is too peaked, so peaked that temperature can't move it. And why is it so confident? Because the corpus is only 304 characters and it has all but memorized them. Chapter 24 will come back to this.

**Knob two: top-k.**

It does a different job from temperature:

```python
keep = np.argsort(-adjusted)[:top_k]     # 只留最大的 k 个
```

*(Only keep the largest k.)*

Fix the temperature at 4.0 and change only k:

```text
    top-k  1  -> 故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月光疑
    top-k  2  -> 故乡波前波是明月歌明月低歌上鹅曲白红
    top-k  3  -> 故乡思头望明月低故故故乡鹅曲项向清故
    top-k  5  -> 故乡头前明鹅地是明月是地歌头故乡鹅曲
    top-k 10  -> 故乡前举鹅鹅歌浮是曲项向明上鹅拨清明
```

*(Same opening. At k=1 it writes the poem text correctly; as k grows the output stays Chinese-looking but scrambles the characters more and more, because it can now draw from a longer tail.)*

`k=1` is greedy.

The two knobs do not control the same thing:

- **Temperature controls "how much risk are you willing to take".** It keeps all 33 candidates, it just redistributes the probability. Turn it up and characters that were nearly impossible can get drawn, and sometimes you get something truly absurd.
- **top-k controls "how many candidates are you allowed to choose among".** It cuts off the tail first, and what's left is still split in the original proportions. So it gives you variety without writing anything too weird.

## The new mechanism

This chapter has two new things: one action, and two knobs.

**The action: autoregression.**

```python
ids = list(tokenizer.encode(prompt))
for _ in range(length):
    window = np.array(ids[-CONTEXT:])[None, :]
    logits = model(window).data[0, -1]
    ids.append(pick_next_token(...))          # 挑一个，接到后面
return tokenizer.decode(ids)
```

*(The comment reads "pick one and append it".)*

Four lines. The whole of "generation" is these four lines.

Why does it work? Because the model was trained to "given the preceding text, output a distribution over the next token" (Chapter 20). So as long as you give it some preceding text, it can give you the next cell. You append that cell, the preceding text gets longer, and it can give you one more cell.

**"Generation" is not a capability of the model; it is a loop.** The model only predicts one cell. The loop is what turns one cell into a passage.

**Knob one: temperature.**

```python
adjusted = probabilities ** (1.0 / temperature)
adjusted = adjusted / adjusted.sum()
```

**Knob two: top-k.**

```python
keep = np.argsort(-adjusted)[:top_k]
```

Almost every language model today ships with these two knobs. The temperature slider you see in a web interface is the first one.

## Python implementation

The three strategies actually share the same piece of code; only the parameters differ:

```python
def pick_next_token(probabilities, temperature=1.0, top_k=None, rng=None):
    # 1. 温度：重新分配概率
    adjusted = probabilities ** (1.0 / temperature)
    adjusted = adjusted / adjusted.sum()

    # 2. top-k：砍掉尾巴
    if top_k is not None and top_k < len(adjusted):
        keep = np.argsort(-adjusted)[:top_k]
        cut = np.zeros_like(adjusted)
        cut[keep] = adjusted[keep]
        adjusted = cut / cut.sum()

    # 3. 抽一次
    return int(rng.choice(len(adjusted), p=adjusted))
```

*(The three comments: "1. temperature: redistribute the probabilities", "2. top-k: cut off the tail", "3. draw once".)*

So:

| Strategy | How to set it |
|---|---|
| greedy | `top_k=1` |
| temperature | pass only `temperature` |
| top-k | pass both |

`after.py` runs them side by side:

```text
  同一个开头：「举头望明月低头思」

  贪心（每次都取最大）
    第 1 次：故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月
    第 2 次：故乡鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波床前明月

  温度 4.0
    第 1 次：故乡明思望明月低思是明光疑前明故前故乡头乡地歌明
    第 2 次：故浮望明月低头思故乡绿明月头思故乡头思故白毛疑霜

  top-k 3 + 温度 4.0
    第 1 次：故乡波前波是明月歌明月低歌上鹅曲曲曲曲曲向拨明月
    第 2 次：故望明月低头曲项向天歌毛拨毛疑是掌拨波床月月月月
```

*(Same opening throughout. Greedy: two runs, both identical poem text. Temperature 4.0: two runs, two different strings — the first one starts 故乡 then scatters through corpus characters; the second keeps more poem-like fragments. top-k 3 + temperature 4.0: two more different strings, still visibly made of poem characters but scrambled.)*

The bottom two groups differ between runs. The first group is identical both times.

## What it solves

The model itself hasn't changed by a single character; the parameter count is the same, not one weight moved. The only thing that changed is "what we do with those 33 probabilities once we have them".

But the capability we get is completely different:

**From "predict" to "generate".** In Chapter 22 we said "it only predicts, it does not generate". That sentence can be retired now: chain the predictions together and you have generation. The act of chaining is called autoregression, and the loop is what gives it this capability.

**From "one path" to "a field of paths".** Greedy only ever walks one path. Now we can pick among a whole crowd of possible continuations. Switch to the opening `床前` and look at the last section of `experiment.py`:

```text
  贪心
    第 1 次：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿
    第 2 次：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿
    第 3 次：床前明月光疑是地上霜举头望明月低头思故乡鹅鹅鹅曲项向天歌白毛浮绿

  温度 4
    第 1 次：床前明月低上鹅月地思故故乡向天地歌明歌歌白歌举头明明光白鹅鹅曲掌
    第 2 次：床前明月光疑是地地上霜举头望明月低头乡波头思明月低头望故前故低水
    第 3 次：床前明月光地光是地上霜举头望明月低头歌思故乡鹅曲项向天白毛浮毛歌
```

*(Opening: 床前, "before my bed". Greedy: three runs, three identical poem continuations. Temperature 4: three different strings — each one starts out poem-like (床前明月光疑是地上霜 in runs 2 and 3) and then drifts, repeating characters and wandering into the goose poem.)*

Same model, same opening. The first setting gives the same thing every time; the other two give something different every time.

**And "how different" is adjustable.** Temperature and top-k are those two knobs.

There is one more number in `experiment.py` worth mentioning. It measures the "longest repeat" — the longest common substring between what was written and the corpus:

```text
    2.0        1.11             6        32
    3.0        1.62             6        24
    4.0        2.77             6        12
    6.0        6.98             6        11
    8.0       12.09             6         9
```

*(Same four columns as before: temperature, effective candidates, distinct outputs in 6 runs, longest repeat.)*

The higher the temperature, the shorter the longest repeat. That is, **letting the model be a bit more "free" makes it look less like it's reciting from memory**.

Careful, though: at temperature 8 it no longer looks much like Chinese. We made a trade between "looks like Chinese" and "has variety".

## What it still can't solve

In this chapter we got two knobs, and we worked out what the word "generation" actually refers to.

But there's one question we've been dragging along since Chapter 22 without answering head-on:

**What does this model actually know?**

The only thing we know is that the characters it writes "look like Chinese". But "look like" is very weak evidence.

Specifically, there are three questions we haven't answered any one of:

1. How much of what it writes did it **memorize**? We repeated the corpus 8 times, so it could perfectly well have all 304 characters stored. If that's what's happening, it isn't writing poetry, it's a parrot.
2. Give it a stretch of text from **outside the corpus** (but using characters in the vocabulary) — what does it do? In `before.py` we already saw it collapse, but is that greedy's fault or the model's?
3. Our corpus has no 亮 (the character in 月亮, "moon"; we do have 月 and 光, "moon" and "light"), so it doesn't know the word 月亮. Does it understand that 明月 ("bright moon") and 月光 ("moonlight") are the same sort of thing? Did it learn what characters 月 should hang around with?

And there's a more basic question: **temperature has to be turned to 3 before anything moves, because this model is too confident. Why is it so confident?**

Next chapter we add no new parts and change no architecture. We just run one set of probe experiments, to go and see what it has inside and what it's missing.

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Set `temperature` to `0.01` and to `100`, and run `generate`. What happens in each case? Why must temperature be greater than 0? (`pick_next_token` will raise an error outright — work out why it *should* raise an error.)
2. In `after.py`, `greedy` uses `top_k=1`. If you change it to `temperature=0.0001` (without setting top_k), is the effect the same? Why? (Hint: what is `0.9999 ** 10000`?)
3. In `experiment.py`'s temperature table, add a row for `temperature=20`. How far does the longest repeat fall? Does what it writes still look like Chinese? Find the dividing line between "still readable" and "not readable".
4. Fix `top_k` at 3 and turn the temperature from 0.2 up to 8. You'll find the effective number of candidates maxes out at 3. Why?
