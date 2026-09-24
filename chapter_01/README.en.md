**English** | [中文](README.md)

# Chapter 1: First, split the sentence apart

**A note on the Chinese.** The sentences and the vocabulary in this book are Chinese,
because the book was written in Chinese and the code is frozen — the programs have
`苹果发布新手机` hard-coded in them and print Chinese back at you. Every Chinese word that
matters gets an English gloss the first time it appears, and every piece of program output
is quoted exactly as it came out of the program. None of the mechanics depend on the
language: splitting a sentence apart, and everything we build on top of it later, works
exactly the same way on English text.

## The problem in this chapter

Let's not think about "classification" yet, and let's not think about "learning" either.
Let's ask a dumber question first:

**When the computer is handed 「苹果发布新手机」 ("Apple shipped a new phone"), what does it
actually see?**

Open up Python and take a look:

```python
sentence = "苹果发布新手机"
print(len(sentence))                            # 7
print(sentence[0], sentence[1], sentence[2])    # 苹 果 发
print(sentence == "苹果发布新手机")               # True
print(sentence == "苹果发布了新手机")             # False
```

```text
7
苹 果 发
True
False
```

*(It prints the length, the first three characters, then two string comparisons.)*

That is the whole truth: **to the computer, 「苹果发布新手机」 is 7 characters lined up in a
single string.**

It can count 7, it can pull out the character at position 0 (苹), it can compare two strings
to see whether they are exactly alike — that's all.

It does not know that 苹果 ("apple") is a thing. As far as it's concerned, the relationship
between 苹 and 果 is exactly the same as the relationship between 苹 and 机: both are just
"two characters sitting next to each other".

And units like 苹果 and 发布 ("to release / to ship") are precisely what we want to talk
about. So the first question is:

> **How do we turn a sentence into units the program can actually operate on?**

## The simplest attempt

The most intuitive move is: don't split.

A sentence is a string, so let's just use the whole string as a key and write it all down.

```python
KNOWN_SENTENCES = [
    "苹果发布新手机",
    "苹果发布新芯片",
    # ... the other 8 sentences in the corpus
]

def has_seen(sentence):
    return sentence in KNOWN_SENTENCES
```

This is the smallest program we can possibly write: one list, one `in` check.

How it behaves (`before.py`):

```text
语料里的 10 句原话：
  [听过] 苹果发布新手机
  [听过] 苹果发布新芯片
  ...

同样意思、只是换个写法的 5 句话：
  [没听过] 苹果发布了新手机
  [没听过] 苹果发布新手机。
  [没听过] 苹果发布新手机！
  [没听过] 苹果 发布 新手机
  [没听过] 苹果即将发布新手机

------------------------------------------------------------
原话命中：10/10
换个写法命中：0/5
------------------------------------------------------------
```

*(`[听过]` means "seen before", `[没听过]` means "never seen". All 10 corpus sentences hit;
all 5 reworded sentences miss.)*

Against the original sentences, 100%. Against a reworded version, 0%.

This is not "the program is too dumb". It is that **it has no units to operate on at all.**
The whole sentence is one indivisible atom; if even one character differs, it's a different
atom. One extra 了, one extra space, and it counts as a different sentence.

## Experiment

Run `experiment.py`. We do three things.

**Experiment 1**: confirm the failure above (10/10 on the originals, 0/5 on the reworded
versions).

**Experiment 2**: split the sentences apart and see what they turn into.

```text
============================================================
实验二：拆开之后，一句话变成了什么
============================================================
  苹果发布新手机
      苹果 | 发布 | 新 | 手机
      [0, 1, 2, 3]
  苹果发布了新手机
      苹果 | 发布 | <UNK> | 新 | 手机
      [0, 1, 16, 2, 3]
  苹果芯片很强
      苹果 | 芯片 | 很 | <UNK>
      [0, 4, 9, 16]
  这个苹果真甜
      这个 | 苹果 | 真 | 甜
      [12, 0, 13, 10]
  苹果配香蕉
      苹果 | <UNK> | 香蕉
      [0, 16, 11]

  说明：词表里没有的块（'了'、'强'、空格……）都换成了 <UNK>。
```

*(Experiment 2: after splitting, what a sentence turns into. Each sentence, then the pieces
it splits into, then the ids of those pieces. The last line notes that any piece not in the
vocabulary — 了, 强, the space — is replaced with `<UNK>`.)*

**Experiment 3**: once a sentence is split apart, "how alike" two sentences are can be
counted for the first time. Take 「苹果发布了新手机」 and compare it against every sentence in
the corpus:

```text
    苹果发布新手机     共享 4 个：苹果 发布 新 手机
    苹果发布新芯片     共享 3 个：苹果 发布 新
    华为发布新电脑     共享 2 个：发布 新
    小米发布新手机     共享 3 个：发布 新 手机
    苹果芯片很强       共享 1 个：苹果
    苹果很好吃         共享 1 个：苹果
    苹果很甜           共享 1 个：苹果
    香蕉很好吃         共享 0 个：
    这个苹果真甜       共享 1 个：苹果
    苹果做成派         共享 1 个：苹果

  5 句「换个写法」的最像的一句：
    苹果发布了新手机         最像 -> 苹果发布新手机   共享 4 个词
    苹果发布新手机。         最像 -> 苹果发布新手机   共享 4 个词
    苹果发布新手机！         最像 -> 苹果发布新手机   共享 4 个词
    苹果 发布 新手机         最像 -> 苹果发布新手机   共享 4 个词
    苹果即将发布新手机       最像 -> 苹果发布新手机   共享 4 个词
```

*(`共享 4 个：苹果 发布 新 手机` reads "shares 4 with it: 苹果 发布 新 手机"; the second half
finds the closest corpus sentence for each of the 5 reworded sentences.)*

The 5 sentences the program "completely failed to recognize" in experiment 1 now all share
4 words with their closest match. The extra 了 no longer turns the sentence into a stranger
— because it has been cut off, and what's left lines up.

## The new mechanism

Only one thing: **split the sentence into words, then turn the words into ids.**

Three steps.

**Step one: fix a vocabulary.**

The vocabulary is the list of "all the words we know". We write this list ourselves, and the
whole book uses this one:

```text
苹果 发布 新 手机 芯片 电脑 华为 小米 好吃 很 甜 香蕉 这个 真 做成 派
```

*(16 words: apple (苹果), release/ship (发布), new (新), phone (手机), chip (芯片), computer
(电脑), Huawei (华为), Xiaomi (小米), delicious (好吃), very (很), sweet (甜), banana (香蕉),
this (这个), really (真), make into (做成), pie (派).)*

16 words in total.

**Step two: use the vocabulary to cut the sentence.**

Scan the sentence from left to right. At every position, **take the longest piece you can
find that is in the vocabulary.**

Take 「苹果芯片很强」 ("Apple's chip is very strong") as the example:

```text
苹果芯片很强
^^                 "苹果" 在词表里 -> 切走，指针往后挪 2 格
  苹果芯片很强
  ^^^^             "芯片" 在词表里 -> 切走，指针往后挪 2 格
      苹果芯片很强
      ^^           "很强" 不在词表里，"很" 在 -> 切走，指针往后挪 1 格
        苹果芯片很强
        ^^         "很强" 不在，"强" 也不在 -> 词表里没有这个字
```

*(`^^` marks what is being looked at. The text on the right says whether that piece is in
the vocabulary, and how far the pointer moves.)*

What about that last 强? It isn't in the vocabulary, but it is a character that really exists
in the sentence, and we can't pretend not to see it. We give it one uniform placeholder:
`<UNK>` (unknown).

**Step three: give every word an id.**

A program only does arithmetic; the only string operation it can do is "are these two strings
equal". So every word needs an id. The id is its position in the vocabulary:

```python
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
UNKNOWN_ID = len(VOCABULARY)        # <UNK> goes after all the words
```

That id is the word's token id. Every word in the vocabulary has one, and `<UNK>` has one
too.

So 「苹果发布新手机」 becomes:

```text
苹果 | 发布 | 新 | 手机        ->        [0, 1, 2, 3]
```

*(The four words on the left, the four ids on the right.)*

A string of numbers. At last the program has something it can operate on.

## Python implementation

The full code is in `after.py`. The core is these two functions.

```python
def split_sentence(sentence):
    words = []
    position = 0
    while position < len(sentence):
        matched_word = None
        for length in range(MAX_WORD_LENGTH, 0, -1):
            piece = sentence[position:position + length]
            if piece in WORD_TO_ID:
                matched_word = piece
                break
        if matched_word is None:
            words.append(UNKNOWN_WORD)
            position += 1
        else:
            words.append(matched_word)
            position += len(matched_word)
    return words
```

A few things worth noticing:

- `for length in range(MAX_WORD_LENGTH, 0, -1)`: try the longest first (2 characters), then
  shorter (1 character). This is "longest match". Without it, 苹果 would be cut into 苹 and
  果 — and neither of those characters is in the vocabulary at all.
- When we cut a word, `position += len(matched_word)`: however many characters we took, that's
  how far the pointer moves.
- **When we can't cut a word, `position += 1`**: this line is very easy to get wrong. If we
  leave the pointer sitting there when nothing matches, the program spins in place forever.
  And moving exactly 1 is what guarantees that every character of the original sentence gets
  processed once and not one is dropped — `test_chapter_01.py` tests exactly this.

```python
def to_ids(words):
    return [WORD_TO_ID.get(word, UNKNOWN_ID) for word in words]
```

`to_ids` turns words into ids. `WORD_TO_ID.get(word, UNKNOWN_ID)` means: if the word is in
the vocabulary, use its id; if not, use the id of `<UNK>`. We deliberately don't write
`try/except` — when the lookup fails, `get`'s default value takes over, because "I don't know
this word" is a normal situation, not an error.

## What it solves

This chapter did exactly one thing: **it let the program see the "pieces" inside a sentence
for the first time.**

Let the data from experiment 3 make the case. 「苹果发布了新手机」, with the extra 了, gets 0
hits when the whole sentence is compared; once it's split apart, it shares 4 words with
「苹果发布新手机」 from the corpus — 苹果, 发布, 新, 手机, all four of them there. The extra
了 gets cut out as a single `<UNK>`, and it no longer pollutes the rest.

Look at one more sentence from experiment 2: `苹果配香蕉 -> [0, 16, 11]`. 配 ("to pair with")
has never appeared in the corpus, and the program still turned it into an id. Whole-sentence
comparison can't do that — it has no way to deal with a character it has never seen, because
the unit it compares is the entire string, not one piece at a time.

Concretely, the capability looks like this:

| before | now |
|---|---|
| the unit of a sentence is "the whole string" | the unit of a sentence is "one word at a time" |
| the only comparison is "equal / not equal" | every sentence is a list of ids, and we can look at it position by position |
| one extra character = nothing matches at all | one extra character = one extra `<UNK>` |

## What it still can't solve

The program can now say that 「苹果发布了新手机」 contains the pieces `[0, 1, 16, 2, 3]`.

But it has no idea what those pieces mean.

Hand it the 10 corpus sentences and all it sees is 10 strings of ids. It won't know that the
sentences carrying 手机 ("phone"), 芯片 ("chip"), 电脑 ("computer") are one group, and the ones
carrying 好吃 ("delicious"), 甜 ("sweet"), 香蕉 ("banana") are another group. **The difference
between those two groups simply does not exist as far as the current program is concerned.**

So:

> **The words are split out. How do we know what category a sentence belongs to?**

Next chapter we'll try to make it answer that question. The approach will be pretty dumb —
every single condition will be hand-written by us.

## Exercises

See `exercises.en.md`. The 4 most important ones:

1. Force `MAX_WORD_LENGTH` to 1, run `after.py`, and see what 「苹果发布新手机」 gets cut into.
   Then think it through: why is "longest match" necessary rather than optional?
2. Put a rare character that isn't in the vocabulary into a sentence (say 「苹果很好吃呀」,
   "apples are really yummy"), confirm that it turns into one `<UNK>` and doesn't affect the
   other words. Then take it one step further: what if the whole sentence is rare characters?
3. The current rule is "take the longest piece you can". Find a sentence where the
   vocabulary's words admit two different cuts, and say what's wrong with each of them.
4. Where does the vocabulary come from? This 16-word vocabulary was **hand-written by us**.
   What if we replaced it with a dictionary (tens of thousands of words)? Would hand-writing
   still work?
