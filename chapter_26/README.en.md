**English** | [中文](README.md)

# Chapter 26: A real tokenizer

## The problem in this chapter

Chapter 25 ended with a sentence: **our tokenization is too crude.**

Look back at what we have been doing all along: one character, one token. That served us
well for the first 24 chapters, because it was simple enough to need no explanation. But
it has two flaws we can't dodge.

The first flaw is about the vocabulary. We tried Chapter 1's hand-written vocabulary on
`苹果不好吃` ("apples are not delicious"):

```text
苹果不好吃 -> ['苹果', '<UNK>', '好吃']   <- 有生词！
```

*(The arrow shows the resulting tokens: `苹果` "apple", `<UNK>` for a word the vocabulary
doesn't contain, `好吃` "delicious". One word is unknown.)*

A single character, `不` ("not"), flips the meaning of the whole sentence, and it does not
exist in the vocabulary at all. What is more unsettling is that even Chapter 1's own
sentences can't escape it:

```text
苹果芯片很强 -> ['苹果', '芯片', '很', '<UNK>']  <- 有生词！
```

*(`苹果芯片很强`, "the Apple chip is very strong" → `苹果` "Apple", `芯片` "chip", `很`
"very", and `<UNK>` — the vocabulary has no `强` "strong".)*

The vocabulary has `很`, it has `甜`, it has `好吃`, but it does not have `强`. Because
**this table was written by a person, and a person can't write all of it**.

The second flaw: what if we want to write a more complete one?

```text
  我们的语料一共 5407 个字。
  语料里不同的 1 字片段：50 个
  语料里不同的 2 字片段：88 个
  语料里不同的 3 字片段：141 个
  语料里不同的 4 字片段：191 个
  语料里不同的 5 字片段：298 个

  语料越大，新片段就越多，词表跟着一直长。理论上的上限更吓人：
  常用汉字 3000 个，1 字组合最多有 3,000个
  常用汉字 3000 个，2 字组合最多有 900 万个
  常用汉字 3000 个，3 字组合最多有 270 亿个
  常用汉字 3000 个，4 字组合最多有 81 万亿个
```

*(The corpus has 5,407 characters in total. The distinct 1-character fragments number 50;
2-character fragments, 88; 3-character, 141; 4-character, 191; 5-character, 298. The
bigger the corpus, the more new fragments appear and the longer the vocabulary grows. The
upper bound is worse: with 3,000 common Chinese characters there can be up to 3,000
1-character combinations, 9 million 2-character ones, 27 billion 3-character ones, and
81 trillion 4-character ones.)*

Chapter 2's nightmare of "tens of thousands of ifs" is back, wearing a different coat.

**And while we're at it, let's answer a question you may have been wanting to ask: why
did Chapter 1 only teach "split the sentence into words", and never mention BPE, or
subwords, or bytes?**

Because Chapter 1's you hadn't run into these two problems yet. Back then the corpus was
ten sentences and the vocabulary was 16 words, and "split into words" was the only thing
that couldn't go wrong. **A mechanism is only easy to understand after its pain has
appeared.** If someone had opened with "BPE is a subword tokenization algorithm", you
would have remembered the name without knowing what fire it was putting out. Now you know
what it has to put out: unknown words, and a vocabulary explosion.

## The simplest attempt

The most intuitive fix: write a bigger vocabulary.

That is also what the earliest Chinese tokenizers did — find a dictionary, collect a few
hundred thousand words, and use it to cut sentences into words (`word_tokenize` in
`before.py` does exactly this: starting from the current position, match the longest word
it can).

It really is better than the character level: fewer tokens, and the meaning stays whole —
`苹果很甜` ("apples are very sweet") is only 3 tokens. But the dictionary road has a dead
end:

- A dictionary always has fewer words than the language. New words appear every day
  (`直播带货` "livestream selling", `折叠屏` "folding screen"); the dictionary can't keep
  up, and even when it does it can't cover technical terms, names, typos, or emoji.
- The bigger the dictionary, the more an unknown word costs you: one character that isn't
  in the dictionary drags down the whole sentence.
- And the dictionary is **decided by people**. We want a vocabulary that **grows out of
  the corpus by itself**.

So is there a way to **decide the vocabulary size ourselves, without writing a
dictionary, and without ever having an unknown word?**

## Experiment

`after.py` has a working mini-BPE, and `experiment.py` puts it side by side with three
other ways of splitting text.

### Four ways of splitting, on the same set of sentences

```text
苹果不好吃
  字符级    5 个  ['苹', '果', '不', '好', '吃']
  词级      3 个  ['苹果', '<UNK>', '好吃']   <- 1 个生词
  BPE       3 个  ['苹果', '不', '好吃']
  字节版    5 个  ['苹果', '�', '�', '�', '��吃']

苹果很甜
  字符级    4 个  ['苹', '果', '很', '甜']
  词级      3 个  ['苹果', '很', '甜']
  BPE       2 个  ['苹果', '很甜']
  字节版    1 个  ['苹果很甜']

华为发布新平板
  字符级    7 个  ['华', '为', '发', '布', '新', '平', '板']
  词级      5 个  ['华为', '发布', '新', '<UNK>', '<UNK>']   <- 2 个生词
  BPE       4 个  ['华为', '发布新', '平', '板']
  字节版    8 个  ['华为', '发布新', '�', '�', '�', '�', '�', '�']

我昨天在商场看到苹果刚刚发布的新手机
  字符级   18 个  ['我', '昨', '天', '在', '商', '场', '看', '到', '苹', '果', '刚', '刚', '发', '布', '的', '新', '手', '机']
  词级     15 个  ['<UNK>', ..., '苹果', '<UNK>', '<UNK>', '发布', '<UNK>', '新', '手机']   <- 11 个生词
  BPE       2 个  ['我昨天在商场看到', '苹果刚刚发布的新手机']
  字节版    7 个  ['我昨天在商场看到', '苹果', '刚', '刚', '发布', '的新', '手机']

（字节版里的 '�' 是半个字的字节：那个字在训练语料里没出现过，
  所以 BPE 还没学会把它粘起来。字节版永远切得开，但可能切得难看。）
```

*(The four rows under each sentence are character-level, word-level, BPE, and the
byte-level version, with the token count and the tokens for each. Read them as: character
level splits every sentence into single characters; word level uses Chapter 1's
hand-written vocabulary and produces `<UNK>` for anything missing (one unknown word in
`苹果不好吃`, two in `华为发布新平板`, eleven in the long sentence); BPE learns its own
fragments; the byte-level version works on bytes, so characters it never saw during
training come out as half-character `�` pieces.)*

Notice the two long sentences: word-level tokenization spits out 11 `<UNK>`s on them,
because the function words in that sentence (`我` "I", `昨天` "yesterday", `在` "at")
are nowhere in Chapter 1's table. **Our corpus kept getting more complex, while the
hand-written vocabulary stayed exactly where it was.**

### The whole life story of a BPE

When BPE finishes training it leaves behind a record of how it grew up. The first 20
merges look like this:

```text
前 20 次合并（学到的第一批 token）：
  苹果  发布  。苹果  。小  手机  发布新  很强  好吃  华为  芯片  。小米  这个  这个苹果  。这个苹果  。华为  电脑  香蕉  。香蕉  考试  给了
```

*(The first 20 merges — the first batch of tokens it learned.)*

Merge 40 more times and longer fragments start appearing:

```text
第 40-60 次合并（开始出现更长的片段）：
  发布的新手机  刚发布的新手机  刚刚发布的新手机  。小王把书给了小李因为他明天考试  。我昨天在商场看到  很好吃  真甜  成派  做成派  。小米发布新  很甜  。苹果发布新  芯片很强  真好吃  。华为发布新  电脑很强  手机很强  。苹果很好吃  苹果刚刚发布的新手机  。我昨天在商场看到苹果刚刚发布的新手机
```

*(Merges 40–60 — fragments start getting longer.)*

### BPE has exactly one knob

```text
合并次数    词表大小    语料总 token 数   这句话几个 token
------------------------------------------------------------------------
0           50          5,407             18
50          100         1,494             3
100         150         580               2
200         250         379               2
500         550         79                2

合并得越多：词表越大、序列越短。
词表大 = 输出层的矩阵大；序列短 = 模型要算的位置少。这就是分词要权衡的东西。
```

*(Rows: number of merges, vocabulary size, total tokens in the corpus, and how many
tokens that one long sentence becomes. More merges means a bigger vocabulary and shorter
sequences.)*

The more you merge: the bigger the vocabulary, the shorter the sequence.
A big vocabulary = a big output-layer matrix; a short sequence = fewer positions for the
model to compute. That is the trade-off tokenization has to make.

### Nobody told it what a "word" is

```text
第 1 章那张词表里，长度大于 1 的词有 11 个：苹果  发布  手机  芯片  电脑  华为  小米  好吃  香蕉  这个  做成
其中被 BPE 自己合并出来的：9 个 -> 苹果  发布  手机  芯片  电脑  华为  好吃  香蕉  这个

没有人告诉 BPE 中文的词长什么样。它只知道'哪两个 token 老挨在一起'。
结果它把'苹果''发布''手机''好吃'这些片段拼了出来 —— 因为它们确实老挨在一起。
```

*(Of the 11 multi-character words in Chapter 1's vocabulary, BPE merged 9 of them by
itself: `苹果` `发布` `手机` `芯片` `电脑` `华为` `好吃` `香蕉` `这个`. Nobody told BPE what
a Chinese word looks like — it only knows which two tokens keep ending up next to each
other.)*

Out of those 11 words, it assembled 9 on its own. The two it missed (`做成` "made into",
`这个` "this") are the ones that don't appear often enough in our corpus — BPE only
counts frequency, it doesn't know grammar.

## The new mechanism

**BPE (Byte Pair Encoding).** The name sounds strange (it started life as an algorithm in
a 1994 data-compression paper), but what it does fits in one sentence:

> At the start, every character is a token.
> Then do one thing over and over: **find the two tokens that are most often next to each
> other right now, and glue them into a new token.**

Each glue adds one token to the vocabulary. How many times we glue is up to us.

```
How big a vocabulary do I want?
    vocabulary size = the number of characters at the start + the number of merges

So: merge 200 times, and the vocabulary is 50 + 200 = 250 tokens.
```

As pseudocode it is three lines:

```python
for _ in range(number_of_merges):
    find the adjacent pair (a, b) that occurs most often
    replace every a b in the sequence with ab
```

**Why does this solve the two problems above?**

- **No unknown words.** The worst case is "not a single merge happened", which falls back
  to individual characters. And the character table was counted out of the corpus, so it
  is **guaranteed to be enough**.
- **A controllable vocabulary size.** The vocabulary is no longer decided by "how many
  words the language has", but by "how many times we feel like merging". Want a
  50,000-token vocabulary? Merge 50,000 times. Want it smaller? Merge fewer times.

**The BPE family tree.** At this point we can line up the four ways of doing tokenization:

| Approach | Smallest unit | Where the vocabulary comes from | Unknown words | Sequence length |
|---|---|---|---|---|
| Character level | characters | characters seen in the corpus | none (but every character must be learned separately) | longest |
| Word level | words | a dictionary people wrote | yes, `<UNK>` | shortest |
| **BPE** | characters → merged fragments | merged out of the corpus | none | medium |
| **byte-level BPE** | bytes (0–255) | merged out of 256 bytes | none, any text can be encoded | on the long side for Chinese |

The last one deserves a section of its own. Models after GPT-2 use the byte version: it
**doesn't even presuppose the concept of a "character"**, it just treats text as a string
of bytes from 0 to 255. The advantage is that anything at all (traditional characters,
rare characters, emoji, code, binary noise) can be encoded, and there is never an
`<UNK>`. The cost is that one Chinese character takes 3 bytes, so part of the merge
budget has to go into gluing "the three bytes of one character" back together:

```text
合并次数      字符版 token 数       字节版 token 数       
----------------------------------------------------------------
0         7                 21                
50        3                 4                 
200       3                 3                 
```

*(Rows: number of merges, tokens for the character version, tokens for the byte version.
At 0 merges the byte version is three times as long; once you merge enough, it catches
up.)*

Look at the first row: before any merging, the byte version is 3 times as long as the
character version. After enough merging, it catches back up.

(Implementation-wise, the byte version and the character version **are the same code**;
all that changes is what the "smallest unit" is — `train_bpe(text, times, unit=to_bytes)`
in `after.py` switches it over.)

## Python implementation

`after.py` has three functions in total, and that is all of BPE.

**Step one: count adjacent pairs.**

```python
def count_pairs(tokens):
    """Count how many times each adjacent pair of tokens occurs together."""
    pairs = {}
    for left, right in zip(tokens, tokens[1:]):
        pairs[(left, right)] = pairs.get((left, right), 0) + 1
    return pairs
```

**Step two: merge.**

```python
def merge_pair(tokens, pair):
    """Merge every occurrence of `pair` in the sequence into one token: A B -> AB."""
    merged = []
    position = 0
    while position < len(tokens):
        if (position < len(tokens) - 1
                and tokens[position] == pair[0] and tokens[position + 1] == pair[1]):
            merged.append(pair[0] + pair[1])
            position += 2
        else:
            merged.append(tokens[position])
            position += 1
    return merged
```

**Step three: the loop.**

```python
def train_bpe(text, num_merges, unit=None):
    tokens = unit(text) if unit else list(text)
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(tokens)
        if not pairs:
            break
        best = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best)
        tokens = merge_pair(tokens, best)
    return merges
```

That `merges` list is the entire product of training: **an ordered list of merge rules**.
It is also the vocabulary — the new token produced by rule *i* is the *i*-th new member of
the vocabulary.

**Encoding** is just applying that list of rules in order:

```python
def encode(text, merges, unit=None):
    tokens = unit(text) if unit else list(text)
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens
```

Why does applying them once, in order, equal "merging all over again"? Because a rule can
only ever appear in a context created by the rules before it — so starting from the first
one is guaranteed to be right.

**Decoding** is even simpler: every merge only ever glued two tokens together, so joining
them back up restores the original.

```python
def decode(tokens):
    return "".join(tokens)
```

The property `decode(encode("苹果发布新手机")) == "苹果发布新手机"` is one the tests must
pass.

## What it solves

- **The vocabulary size becomes adjustable.** Merge 200 times → 250 tokens; merge 500
  times → 550 tokens. We no longer need to hand-write a dictionary.
- **The unknown-word problem disappears.** `苹果不好吃` is `['苹果', '<UNK>', '好吃']`
  under word-level tokenization and `['苹果', '不', '好吃']` under BPE — the character
  `不` never appeared even once in the training corpus, and BPE still cut it out, because
  in the worst case it can always fall back to characters.
- **Sequences get shorter.** That 18-character sentence goes from 18 tokens to 2 tokens
  (same text, an order of magnitude fewer positions for the model to compute). Half of
  why the Chapter 22 model couldn't handle long sentences was that tokenization made them
  too long.
- **Nobody told it what a word is**, and it assembled `苹果`, `发布`, `手机`, `好吃` out
  of frequencies by itself (of the 11 words in Chapter 1's table, it grew 9 on its own).

## What it still can't solve

That more or less wraps up tokenization. But notice this: **BPE only turns text into a
string of ids. It does not make the model one bit smarter.**

Now put our model next to GPT and go item by item:

| | Our model (Ch. 22/25) | GPT |
|---|---|---|
| Tokenization | character level (just upgraded to BPE in this chapter, but the model doesn't use it yet) | byte-level BPE, 50,000 tokens |
| Normalization in each block | compute, add, then LayerNorm | LayerNorm, then compute |
| MLP activation | ReLU | GELU |
| Position information | the fixed sinusoidal encoding from Ch. 16 | a learnable position table |
| Layers × dimension | 2 layers × 32 | 96 layers × 12288 |
| Parameters | 27,812 | 175 billion |

In that right-hand column, there are a few entries we have never written: LayerNorm before
computing, GELU, a learnable position table. Are they some new mechanism we don't know
about yet? Or old things used in a new way?

The next question is: **is there any module in GPT that we haven't written?**

## Exercises

See `exercises.en.md`; here are the 3 most important ones:

1. Change `train_bpe`'s merge count from 200 to 50 and to 500, and see how the split of
   `苹果不好吃` and the total token count of the corpus change. How big does the
   vocabulary become?
2. Manually add `不`, `强`, `平`, `板` to `word_tokenize` and re-run `before.py`. Are
   there fewer unknown words? And how do you plan to guarantee there won't be new ones
   next time?
3. After re-tokenizing with BPE, train the Chapter 25 model again and see whether the
   loss goes lower in the same number of steps (hint: `block_size` can now cover longer
   text).
