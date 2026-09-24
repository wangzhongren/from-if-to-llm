**English** | [中文](README.md)

# From If to LLM

### Build a language model with your own hands, starting from a single classification rule

---

## What this book is

Most explanations of the Transformer open like this:

> The Transformer consists of Multi-Head Attention, a Feed-Forward Network, Residual
> Connections, Layer Normalization, and Positional Encoding.

You memorize five terms, and you still don't know why it's those five and not five others.

This book runs the other way.

**Each chapter solves exactly one problem left over by the previous chapter.**

No concept is introduced early. You hit the problem first, then invent the mechanism, and
only then are you told what the mechanism is called.

By the end, the thing you were *forced* to invent turns out to be the Transformer. Nobody
handed you the parts list — you had no choice but to build them, one at a time.

---

## The chain

```text
text
 → split it apart
 → if/else rules
 → turn the rules into numbers
 → let the machine learn the rules
 → learn features
 → learn word representations
 → learn context
 → learn relationships between positions (attention)
 → residuals and normalization
 → the Transformer block
 → predict the next token
 → LLM
```

Every arrow is a "what it still can't solve" from the end of the previous chapter.

---

## A note on the Chinese

This book was written in Chinese, and **its code is frozen** — the English edition adds
documentation only. So the example sentences, the vocabulary, and the training corpus are
all Chinese, and every program prints Chinese.

That is not a problem for the material, and here is why:

- **We gloss the Chinese.** Every word that carries an argument is translated on first
  appearance in a section. Where a program's output *is* the evidence — a weighted table,
  an attention matrix, a generated poem — we explain what it says immediately outside the
  quoted block.
- **Program output is never translated.** Every `text` block in this book is real captured
  output from a real run. Translating it would make the book disagree with the code you
  are about to run. So it stays Chinese, byte for byte.
- **Nothing mechanical depends on the language.** A loss is a loss. An attention head
  computes the same weighted sum over English tokens. What you build here works on any
  language; only the corpus changes.

The running examples early on are sentences like `苹果发布新手机` ("Apple shipped a new
phone") and `苹果很好吃` ("apples are delicious"). The word `苹果` ("apple") appears in
both — deliberately. It comes back in Chapter 12 as the reason a single vector per word
isn't enough.

Later, the character-level language model of Chapters 20–24 trains on a few lines of
classical Chinese poetry, because it is small, public domain, and pretty.

---

## Map of the book

### Part I — Why machines can "understand" text

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [1](chapter_01/README.en.md) | Split a sentence apart | Words are split out — how do we know what category a sentence is? |
| [2](chapter_02/README.en.md) | Teach a machine with if/else | Tens of thousands of words — do we write tens of thousands of `if`s? |
| [3](chapter_03/README.en.md) | Turn `if` into numbers | Who decides these 1.2, 0.8, 1.5 numbers? |
| [4](chapter_04/README.en.md) | Let the program change its own weights | By how much should each one change? |
| [5](chapter_05/README.en.md) | How wrong are we, exactly? | With many parameters, tweaking them one at a time is too slow |
| [6](chapter_06/README.en.md) | How does it know which way to move? | Every layer's gradient is hand-derived; more layers will kill us |

### Part II — From classifier to neural network

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [7](chapter_07/README.en.md) | A straight line isn't enough | We added a middle layer and it did nothing |
| [8](chapter_08/README.en.md) | Why activation functions | Every gradient is still hand-derived |
| [9](chapter_09/README.en.md) | Let the machine compute its own gradients | Words are still just IDs, and IDs carry no meaning |

### Part III — How text becomes "meaning"

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [10](chapter_10/README.en.md) | An ID number means nothing | Where do these numbers come from? Are they random? |
| [11](chapter_11/README.en.md) | Make similar words land close together | One word is allowed only one meaning |
| [12](chapter_12/README.en.md) | One word can't have just one meaning | Averaging is too blunt — not every word matters equally |

### Part IV — Why attention had to appear

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [13](chapter_13/README.en.md) | Not every word matters equally | It only knows "similar", not "what I'm looking for" |
| [14](chapter_14/README.en.md) | What I'm looking for, what you have | One way of looking isn't enough |
| [15](chapter_15/README.en.md) | Why multiple heads | Word order is gone |
| [16](chapter_16/README.en.md) | The model can't tell what comes first | Stack up layers and it stops learning |

### Part V — How the Transformer was forced into existence

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [17](chapter_17/README.en.md) | More layers made it worse | The numerical scale is still a mess |
| [18](chapter_18/README.en.md) | Data getting messier as it travels | Attention only *moves* information; it doesn't *process* it |
| [19](chapter_19/README.en.md) | Why there's an MLP after attention | The output is still a vector, not a word |

### Part VI — From Transformer to language model

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [20](chapter_20/README.en.md) | Bring back the classifier | Training peeks at the answer |
| [21](chapter_21/README.en.md) | Why training can't peek | Time to assemble everything |
| [22](chapter_22/README.en.md) | The first real small language model | It can predict, but it can't generate |
| [23](chapter_23/README.en.md) | Why the model "generates" | What did it actually "understand"? |
| [24](chapter_24/README.en.md) | Why it looks like it's thinking | Why is the toy model so weak? |

### Part VII — From toy model to modern LLM

| Ch. | Title | The problem it leaves behind |
|---|---|---|
| [25](chapter_25/README.en.md) | Rewrite the whole thing in PyTorch | Our tokenization is too crude |
| [26](chapter_26/README.en.md) | A real tokenizer | How else do we differ from GPT? |
| [27](chapter_27/README.en.md) | GPT | Why is it so big? |
| [28](chapter_28/README.en.md) | Why bigger models work better | Predicting the next word isn't chatting |
| [29](chapter_29/README.en.md) | From language model to ChatGPT | — |
| [30](chapter_30/README.en.md) | Back to the first line of code | The end |

---

## How to read it

**Don't skip chapters.** The entire value of this book is the ordering — every chapter is
built on the previous chapter's *defect*. Skip Chapter 12 and Chapter 13's attention
arrives with no reason to exist.

Each chapter has the same eight sections:

| Section | What it does |
|---|---|
| **The problem in this chapter** | What the previous chapter left broken |
| **The simplest attempt** | The most obvious (usually wrong) fix first |
| **Experiment** | Run it, watch the old approach fail |
| **The new mechanism** | Only the one concept this chapter actually needs |
| **Python implementation** | The full code |
| **What it solves** | Backed by the measured numbers |
| **What it still can't solve** | Setting up the next chapter |
| **Exercises** | Change the parameters, the data, the structure |

**Run everything.** You will not learn this by reading code — you learn it by watching it
fail.

---

## How to run it

The project ships with its own virtual environment (Python 3.12 + numpy + pytest + torch):

```bash
cd from-if-to-llm

# run a chapter's result
./.venv/bin/python chapter_03/after.py

# the problem that chapter is about (the old approach)
./.venv/bin/python chapter_03/before.py

# the key experiment
./.venv/bin/python chapter_03/experiment.py

# that chapter's tests
./.venv/bin/python -m pytest chapter_03/tests/ -q
```

Run the whole book:

```bash
./run_all.sh          # all 30 chapters
./run_all.sh 13 16    # only chapters 13 through 16
```

### Which chapters are slow

Almost every script finishes in seconds. These few take noticeably longer — **they are not
hung**:

| Script | Roughly | Why |
|---|---|---|
| `chapter_22/experiment.py` | ~1.5 min | Ablates one part at a time, retraining for each |
| `chapter_24/experiment.py` | ~3 min | Trains seven models for a 2×2 cross-evaluation |
| `chapter_22/after.py` | ~20 s | It really is training the small language model |

Everything else is under about 20 seconds. All randomness is seeded, so the losses and
accuracies you get match the book exactly. Only **wall-clock time and memory** vary by
machine.

### Dependencies

| Dependency | Used in |
|---|---|
| Python standard library | Chapters 1–8 (**deliberately no third-party libraries**, so the mechanism stays visible) |
| numpy | Chapter 9 onward |
| pytest | Tests throughout |
| PyTorch | Chapter 25 onward |

Chapters 1–24 use **no** deep learning framework. Chapter 9 builds our own automatic
differentiation engine ([toygrad/](toygrad/)), and the language model of Chapter 22 runs on
top of it. When you swap it for PyTorch in Chapter 25, you should have the feeling of
*"wait, this is the thing I already wrote"* — which is exactly the point.

To reinstall the environment:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

---

## Chapter layout

```text
chapter_XX/
├── README.md        # the chapter (also README.en.md — you are reading the English edition)
├── before.py        # the previous chapter's code, applied to this chapter's problem — it fails
├── after.py         # this chapter's result — the improvement is visible
├── experiment.py    # makes "why the old way fails / why the new way works" visible
├── exercises.md     # exercises (also exercises.en.md)
└── tests/           # tests for this chapter's mechanism
```

`before.py` is a feature of this book: **it is wrong on purpose.** Watching it fail is what
makes you actually want the next chapter's mechanism.

---

## Shared code

| Directory | What it is |
|---|---|
| [toygrad/](toygrad/) | Chapter 9's product: a 300-line automatic differentiation engine. Chapters 10–24 run on it |

`toygrad` is deliberately slow and verbose — it is written to be understood, not to be
fast.

---

## License

| Content | License | You may |
|---|---|---|
| **Code** (every `.py` file) | [MIT](LICENSE) | Use it, change it, ship it commercially — just keep the copyright notice |
| **Text** (every `.md` file, both editions) | [CC BY 4.0](LICENSE-docs) | Share it, adapt it, translate it, teach with it, including commercially — with attribution |

Want to teach this to your own team, translate it, turn it into videos, or lift parts into
your own material? Go ahead. Just credit it.

The corpus is three Tang-dynasty poems (*Quiet Night Thoughts*, *Ode to the Goose*,
*Spring Dawn*), all long in the public domain. The remaining example sentences
(`苹果发布新手机` and friends) were written for this book.

---

## One last thing

This book holds a single line:

> **Don't tell the reader which components a Transformer has. Put the reader in front of
> problem after problem until they have no choice but to invent it themselves.**

If you reach Chapter 22 and can guess what the next chapter has to add *before* you turn
the page — this book did its job.
