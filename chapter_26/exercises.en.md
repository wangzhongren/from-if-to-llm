**English** | [中文](exercises.md)

# Chapter 26 Exercises

## 1. Turn the BPE knob

Open `after.py` and change `num_merges` from 200 to 50, 500, and 2000, running each one.

Record three things:

| Number of merges | Vocabulary size | Total tokens in the corpus | How `苹果不好吃` is split |
|---|---|---|---|
| 50 | | | |
| 200 | | | |
| 2000 | | | |

Then answer:

- At what merge count does `苹果不好吃` ("apples are not delicious") become 2 tokens? What
  are those 2 tokens?
- What happens if you keep increasing the merges forever? Will the vocabulary keep
  growing? What kind of fragments get merged?

## 2. Patch the vocabulary by hand, then admit defeat

Open `before.py`, add `不`, `强`, `平`, `板` to `VOCABULARY`, and run it again.

- Are there fewer unknown words?
- Now try `苹果不甜` ("apples are not sweet"). What is still missing?
- Can you think of a way to never have to patch it again? (Hint: this is exactly what BPE
  is doing.)

## 3. Why BPE's merge order can't be shuffled

In `after.py`, change `encode` to apply the rules **backwards**:

```python
def encode_wrong(text, merges, unit=None):
    tokens = unit(text) if unit else list(text)
    for pair in reversed(merges):        # applied backwards — what happens?
        tokens = merge_pair(tokens, pair)
    return tokens
```

Run it and compare the results of `encode` and `encode_wrong`:

```python
sentence = "苹果不好吃"
print(encode(sentence, merges))
print(encode_wrong(sentence, merges))
```

Are the results the same? Why are they the same (or why aren't they)?
Then check whether `decode(encode_wrong(...)) == the original text` still holds.

## 4. Run a real byte-level version

GPT-2's approach differs from our byte version in one more way: it first uses a regular
expression to cut the text into "chunks" (an English word, a run of spaces, a piece of
punctuation each become their own chunk), and then runs BPE inside each chunk. That way
`苹果` and ` 苹果` (with a leading space) get different tokens.

Try adding a pre-splitting step before `train_bpe`:

```python
import re
CHUNK = re.compile(r"\s+|[a-zA-Z]+|[0-9]+|.")

def chunked(text):
    return [c for c in CHUNK.findall(text)]
```

(Note that our BPE needs "one char, one token", so after pre-splitting you still have to
break each chunk into characters — but **merging must not cross chunk boundaries**. Think
through how to change the code; this is more trouble than it looks.)

## 5. Count what Chinese costs in tokens

GPT-2's original vocabulary (50,000 tokens) was trained on English, and Chinese gets cut
into very small pieces by it. Measure it with our byte-level BPE:

```python
texts = ["苹果很好吃", "床前明月光疑是地上霜", "hello world", "🍎"]
```

(the first is "apples are delicious", the second is the first half of the poem from
Chapter 20)

- For each kind of text, how many tokens does one character take on average?
- Chinese and English — which is "more expensive", and by how much?
- What does that mean for "the same model's API charges by token"?

## 6. Teach BPE fragments that aren't in the corpus

Add a few sentences containing `不` to `build_corpus` (for example `苹果不甜` "apples are
not sweet", `香蕉不好吃` "bananas are not delicious"), retrain BPE, and see how many
tokens `苹果不好吃` is cut into.

Add a few more? How many does it take before `不好吃` becomes a single token?

This shows one iron law of BPE: **it only merges things it has seen in the data.**
Whether the tokenizer can see a fragment depends entirely on how many times it occurred
in the corpus.
