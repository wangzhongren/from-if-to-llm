**English** | [中文](README.md)

# Chapter 2: Teaching the machine to classify with if/else

## The problem in this chapter

The last chapter left us with a question:

> The words are split out. How do we know what category a sentence belongs to?

First let's be clear about what a "category" is here. We have a batch of sentences, and each
one is already labelled with its category:

```text
科技：苹果发布新手机 / 苹果发布新芯片 / 华为发布新电脑 / 小米发布新手机 / 苹果芯片很强
食品：苹果很好吃 / 苹果很甜 / 香蕉很好吃 / 这个苹果真甜 / 苹果做成派
```

*(科技 = technology, 5 sentences; 食品 = food, 5 sentences.)*

10 sentences in total: 5 technology, 5 food. We want to write a program that takes **one
sentence in and puts one category out.**

What did Chapter 1 give us? Every sentence turned into a string of ids:

```text
苹果发布新手机   -> [0, 1, 2, 3]
苹果发布新芯片   -> [0, 1, 2, 4]
华为发布新电脑   -> [6, 1, 2, 5]
...
苹果很好吃       -> [0, 9, 8]
苹果很甜         -> [0, 9, 10]
```

*(Each line is a sentence followed by its ids; `...` stands for the sentences in between.)*

The ids are there, but looking at these 10 strings of numbers, the program can't see any
difference between the first 5 and the last 5. It doesn't know what `3` (手机, "phone") or `8`
(好吃, "delicious") is supposed to mean — **the ids themselves don't say anything.**

## The simplest attempt

Since the program can already recognize "which sentence this is", the most direct move is:
**memorize the answers along with the sentences.**

```python
KNOWN_LABELS = {
    (0, 1, 2, 3): "科技",
    (0, 1, 2, 4): "科技",
    # ... 10 entries in all
}

def judge_by_lookup(sentence):
    ids = tuple(to_ids(split_sentence(sentence)))
    return KNOWN_LABELS.get(ids, "不知道")
```

(不知道 = "I don't know".)

Run `before.py`:

```text
语料里的 10 句话
  [对] 苹果发布新手机     答：科技
  [对] 苹果发布新芯片     答：科技
  ...
  正确 10/10

语料之外的 4 句话
  [错] 苹果发布了新手机     答：不知道   真实：科技
  [错] 苹果发布新电脑       答：不知道   真实：科技
  [错] 香蕉很甜             答：不知道   真实：食品
  [错] 机器人发布新手机     答：不知道   真实：科技
  正确 0/4
```

*(`对` = right, `错` = wrong, `答` = answer, `真实` = the true label, `正确` = the number
correct. The 10 memorized sentences all come back right; the second half runs 4 sentences from
outside the corpus and it answers "don't know" every time.)*

The 10 memorized sentences are all correct; none of the unseen ones are.

This is not "we didn't memorize enough". Even if we expanded the corpus to 100,000 sentences,
it would be the same: **as long as the pieces in the sentence don't line up, it can't say a
single word.** 「苹果发布新电脑」 is not the same string of ids as anything it memorized, so it
doesn't know this one is technology.

## Experiment

Run `experiment.py`.

**Experiment 1: the rules were patched in, not thought up.**

```text
==================================================================
实验一：补丁史
==================================================================
  第 1 版  只有一条规则  正确  2/10   答错 8 句：苹果发布新芯片、华为发布新电脑、苹果芯片很强……
  第 2 版  补上科技词    正确  5/10   答错 5 句：苹果很好吃、苹果很甜、香蕉很好吃……
  第 3 版  补上食品词    正确 10/10   答错 0 句：无

  补到第 3 版才全对。注意每一版之间隔着的不是'思考'，是'跑一遍看看错在哪'。
```

*(A history of patches. v1, with only one rule, gets 2/10 right; v2 patches in the technology
words, 5/10; v3 patches in the food words, 10/10. The closing line reads "only by v3 is it all
correct. Notice that what sits between one version and the next is not 'thinking' — it's
running it once to see where it went wrong.")*

**Experiment 2: not one rule changed — only the positions of two rules swapped.**

```text
  after.py 里的顺序      正确 10/10
  调换两条之后           正确  6/10

  调换之后答错的句子：
    苹果很好吃         答：科技   真实：食品
    苹果很甜           答：科技   真实：食品
    这个苹果真甜       答：科技   真实：食品
    苹果做成派         答：科技   真实：食品
```

*(The order as it stands in `after.py`: 10/10. After swapping two of them: 6/10. Underneath,
the sentences that break.)*

Not one rule added, not one removed — two rules just moved from later to earlier. 10/10 falls
to 6/10. **Order is part of the rules, too.**

**Experiment 3: test it on sentences from outside the corpus.**

```text
  [错] 苹果好香             答：科技     真实：食品
  [对] 电脑好贵             答：科技     真实：科技
  [对] 香蕉派很好吃         答：食品     真实：食品
  [错] 苹果配奶油           答：科技     真实：食品
  [对] 机器人发布新手机     答：科技     真实：科技
  正确 3/5
```

*(苹果好香 = "the apple smells great", 电脑好贵 = "computers are expensive", 香蕉派很好吃 =
"banana pie is delicious", 苹果配奶油 = "apple with cream", 机器人发布新手机 = "a robot shipped
a new phone".)*

3 out of 5. That looks decent, but the two wrong ones are wrong in completely different ways:

- 「苹果好香」 — 香 ("fragrant") is not in the vocabulary, so the rules can't see it at all;
- 「苹果配奶油」 — not one keyword in it. All that's left is 苹果, and the catch-all rule guessed
  "technology".

## The new mechanism

This chapter has no new formula. It has one shift:

> **Replace "memorize the answers" with "memorize the criteria".**

Memorizing answers means: this sentence is technology, that sentence is food.
Writing rules means: **if the sentence contains 手机, it's technology.**

It looks like nothing but a change of wording, but the nature of it is completely different.
A rule looks at "once the sentence is split apart, is a certain word in there?" And so:

- the sentence is longer or shorter, doesn't matter — we only look at whether the word is there;
- the word order changed, doesn't matter — we only look at whether the word is there;
- **every other word in the sentence is unknown to me**, doesn't matter — as long as my keyword
  is still there.

That's why 「机器人发布新手机」 comes out right in experiment 3: not one of the three characters
of 机器人 ("robot") is in the vocabulary, and all of them turn into `<UNK>` — but the rule looks
at 发布, and that one is still there.

The price is that we have to write the rules ourselves. Written out, they look like this (this
is exactly what's in `after.py`):

```python
def judge(words):
    if "手机" in words:
        return "科技"
    if "芯片" in words:
        return "科技"
    if "电脑" in words:
        return "科技"
    if "华为" in words:
        return "科技"
    if "小米" in words:
        return "科技"
    if "发布" in words:
        return "科技"
    if "好吃" in words:
        return "食品"
    if "很" in words:
        return "食品"      # this one is dangerous, see the note below
    if "甜" in words:
        return "食品"
    # ...
    if "苹果" in words:
        return "科技"      # 苹果 was all we saw, so we guessed
    return "不知道"
```

They are tried from top to bottom, and **whoever is hit first is what comes back.** So the
order does mean something:

- `if "很" in words` is a dangerous rule — 很 ("very") also shows up in 「苹果芯片很强」. It's not
  making mistakes right now purely because the 芯片 rule sits in front of it and catches that
  sentence first.
- `if "苹果" in words` is the catch-all at the bottom of the whole pile. 苹果 shows up in both
  categories, so by itself it can't decide anything. Put it late and the earlier rules get to
  speak first; put it early and it will snatch sentences like 「苹果很甜」 — and that is exactly
  how experiment 2 went from 10/10 to 6/10.

## Python implementation

The full code is in `after.py`. It does two things:

```python
# step one is still the Chapter 1 routine: split the sentence into words
words = split_sentence("苹果芯片很强")     # ['苹果', '芯片', '很', '<UNK>']

# step two: hand it to the rules
judge(words)                              # '科技'
```

`judge` itself has no cleverness in it at all — it's a pile of `if`s tried top to bottom. Only
two spots are worth looking at.

```python
if "很" in words:
    return "食品"
```

This line is the most fragile place in the whole program.

It came in with the third round of patches, alongside 甜, 香蕉 and 派. The thinking at the time
was plain: 很 appears in 「苹果很甜」 and 「苹果很好吃」, so it looks like a clue for the food side,
let's add it.

What we didn't notice when we added it: 「苹果芯片很强」 has a 很 in it too.

The only reason it works is luck plus ordering — the 芯片 rule sits in front of it and catches
that sentence first. The day someone moves the 芯片 rule one slot later, it breaks immediately.

(One more thing: this rule is actually **redundant**. Exercise 2 will have you delete the whole
line and see whether the accuracy changes.)

```python
if "苹果" in words:
    return "科技"
```

This one is even blunter: **we never solved the 苹果 problem, we just guessed.** We guessed
technology because 苹果 shows up a lot in the 5 technology sentences. That's not a judgement,
it's a bet.

As for what this code feels like to read, you'll know the moment you glance at `after.py`:
8 patches, one layer wrapped around another, each with a note after it saying "this one was
patched in". **Its shape is its history.**

## What it solves

For the first time it got the program to answer sentences from outside the corpus:

| | Memorizing sentences (`before.py`) | Writing rules (`after.py`) |
|---|---|---|
| the 10 corpus sentences | 10/10 | 10/10 |
| sentences outside the corpus | 0/4 | 3/5 |

More concretely, it's this one line from experiment 3:

```text
  [对] 机器人发布新手机     答：科技     真实：科技
```

*(It answers "technology" for 「机器人发布新手机」 and the true label is technology — even though
机器人 is not in the vocabulary at all.)*

None of the three characters of 机器人 ("robot") is in the vocabulary; all of them turn into
`<UNK>`. The Chapter 1 program is helpless against a sentence like that — the id sequence
matches no memorized sentence. The rule-writing program gets it right, because the rule only
cares about "is 发布 in there?"

This is the first time, among these 10 sentences, that **a word never seen before appears and
the answer is still correct.**

The price is written plainly in experiment 1: it took us three rounds of patching to reach
10/10.

## What it still can't solve

Look at that table in experiment 1: v1 is 2/10, v2 is 5/10, v3 is 10/10.

Every version's improvement corresponds to "we watched it get something wrong, and then added
a rule."

So how far can this approach stretch?

- The vocabulary is **16 words** right now, and we wrote **8 rules**.
- What if the vocabulary were **50,000 words**? How many rules would we have to write?
- And even if we finished, adding one new word like 香 means changing two places: one word in
  the vocabulary, one rule in the pile. What about 10,000 new words?
- And experiment 2 told us the **order** of these rules still can't be wrong anywhere.

So the question is very concrete:

> **Tens of thousands of words — are we supposed to write tens of thousands of ifs?**

There has to be another way. Next chapter we'll try: **turn the rules into numbers.**

## Exercises

See `exercises.en.md`. The 5 most important ones:

1. Move the `if "很" in words` rule by hand to the very front of `judge`, and run it.
   How far does the accuracy drop from 10/10? Which sentence breaks? Why that one?
2. Delete the `if "很" in words` rule **entirely** and run it again. Did the accuracy change?
   If it didn't, why was that rule ever written?
3. Add a word 香 to the vocabulary, then add a rule that makes 「苹果好香」 answer correctly.
   Count it up: for this one word, how many places did you change?
4. For 「小米很好吃」 ("Xiaomi is delicious"), what do the current rules answer? Is that right?
   小米 and 苹果 ran into the same trouble — say what that trouble is.
5. Which of your rules are really "guesses"? Find them.
