**English** | [中文](exercises.md)

# Chapter 1 exercises

Before you change the code, guess what will happen. Then run it. Guessing wrong teaches you
more than guessing right.

---

## 1. Turn longest match off

In `after.py`, take this line:

```python
MAX_WORD_LENGTH = max(len(word) for word in VOCABULARY)
```

and change it to:

```python
MAX_WORD_LENGTH = 1
```

Run it, and see what 「苹果发布新手机」 gets cut into. The real result is:

```text
苹果发布新手机 -> <UNK> | <UNK> | <UNK> | <UNK> | 新 | <UNK> | <UNK>
```

*(Every character is cut on its own, so 6 of the 7 pieces are `<UNK>`; only 新 survives,
because it is the only single-character word in the vocabulary.)*

Then answer: why is "longest match" not optional, but a necessary rule?

(By the way, think one step further: 「这个苹果真甜」 ("this apple is really sweet") turns into
`<UNK> | <UNK> | <UNK> | <UNK> | 真 | 甜`, but 真 ("really") and 甜 ("sweet") survive. Why do
those two survive?)

---

## 2. Shove a few rare characters in

Add these to `NEW_SENTENCES`:

```python
"苹果很好吃呀",
"苹果发布新手机拉",
"呃",
```

Run it. Confirm:

- 呀, 拉 and 呃 each turn into **one** `<UNK>`, rather than turning the whole sentence into
  `<UNK>`;
- the words around them are still cut out as usual (好吃 is still one word).

Then think: why does `<UNK>` take up only the position of one character, instead of
swallowing the whole sentence?

---

## 3. Work it out by hand

Don't run the program. Work out the ids for these two sentences with pen and paper first,
then check yourself against `after.py`:

```text
香蕉很甜
这个苹果很好吃
```

*(`香蕉很甜` = "bananas are sweet", `这个苹果很好吃` = "this apple is delicious".)*

While you're at it, remember that 苹果 takes up 2 characters and 香蕉 takes up 2 characters,
and keep track of how far the pointer moves each time.

---

## 4. Add one word to the vocabulary

The 强 in 「苹果芯片很强」 is currently an `<UNK>`. Add 强 to `VOCABULARY`:

```python
VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派", "强",
]
```

Run it. Confirm:

- 「苹果芯片很强」 is cut into `苹果 | 芯片 | 很 | 强`, with no `<UNK>` left in it;
- the vocabulary goes from 16 words to 17, and the id of `<UNK>` automatically becomes 17.

There's a trap here. Before the word was added, this sentence's ids were `[0, 4, 9, 16]`;
after the word was added, they are still `[0, 4, 9, 16]`.

**Exactly the same.** But the meaning is completely different:

- before: 16 means "I don't know this word" (`<UNK>`);
- after: 16 means 强.

One number, two meanings. Think it through: if I insert 强 into the **middle** of the
vocabulary (say between 很 and 甜), what happens to the ids of these 10 sentences? What does
that tell you?

---

## 5. Manufacture an ambiguity

Add one more word to the vocabulary: 很甜 ("very sweet"). Now the vocabulary contains 很, 甜
and 很甜 all at once.

Run it, and see what these sentences get cut into (these are the real results):

```text
苹果很甜        -> 苹果 | 很甜
香蕉很甜        -> 香蕉 | 很甜
苹果很好吃      -> 苹果 | 很 | 好吃
这个苹果真甜     -> 这个 | 苹果 | 真 | 甜
```

*(`苹果很甜` = "apples are sweet", `香蕉很甜` = "bananas are sweet", `苹果很好吃` = "apples are
delicious", `这个苹果真甜` = "this apple is really sweet".)*

The same character 甜 ("sweet") sticks to 很 in the first two sentences, but stands on its own
in the fourth.

What this shows is that "how to cut" depends on what's in the vocabulary and what the sentence
looks like — **there is no single correct way to cut.** "Longest match" is just a rule we
made up, not a truth.

Think it through: if we also added 真甜 ("really sweet") to the vocabulary, what would the
fourth sentence become? And then 这个苹果 ("this apple")? What happens if you keep going?

---

## 6. Where did the vocabulary come from?

This chapter's vocabulary is 16 words, and we **hand-wrote** it.

Flip back and look at `VOCABULARY` in `after.py`. Then think:

- If you were doing Chinese sentence classification for real, the vocabulary would be tens of
  thousands of words. Would hand-writing still work?
- If we hit a word that isn't in the vocabulary, our current answer is to throw an `<UNK>` at
  it. If half the words in an article are `<UNK>`, is this program still useful?

(Near the end of this book, we'll come back and deal with "where does the vocabulary come
from".)

---

## 7. Challenge: make it faster

Right now, `split_sentence` tries lengths one at a time starting from the longest, at every
single position.

Add a `print` line to it and count: after cutting all 10 corpus sentences, how many times did
it test `piece in WORD_TO_ID`?

Then think: if the vocabulary had 50,000 words and the sentence had 100 characters, what does
that number become? Is there a way to find "the longest word starting at this position" in
one shot?
