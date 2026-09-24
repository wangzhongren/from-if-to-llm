# 《From If to LLM》English Edition — Translation Spec

> Read this before translating anything. It is the only thing keeping 7 translators
> from producing 7 different books.

---

## 0. The shape of this job

The Chinese edition is finished, verified, and **its code is frozen**. You are producing
English **documentation only**:

```
README.md        →  README.en.md
exercises.md     →  exercises.en.md
```

**Do not create, edit, or rename any `.py` file. Do not touch `toygrad/`.**
Not a single line of code changes. That constraint has a consequence, and it's the
single most important thing in this document — see §2.

---

## 1. The corpus stays Chinese, and that is not a bug

The book's running examples are **Chinese sentences**, and they are **hard-coded in the
programs**:

```python
CORPUS = [("苹果发布新手机", "科技"), ("苹果很好吃", "食品"), ...]
```

You must **not** swap these for English sentences. If the prose says "Apple shipped a new
phone" while the code says `苹果发布新手机` and the program prints `科技`, the reader is
lost. The English edition describes the Chinese edition faithfully.

So: **keep every Chinese word, sentence, and corpus item exactly as-is**, and add a short
English gloss on first appearance in a section. Like this:

> Our ten sentences are things like `苹果发布新手机` ("Apple shipped a new phone") and
> `苹果很好吃` ("apples are delicious").

First appearance in a section gets a gloss; after that, plain Chinese is fine.
The vocabulary table (`苹果 发布 新 手机 芯片 …`) likewise stays Chinese, with a gloss
column or inline glosses.

Add a short **"A note on the Chinese"** paragraph near the top of the book (root README)
and once in Chapter 1 explaining plainly:

- the corpora are Chinese because the book was written in Chinese and the code is frozen;
- every Chinese word that matters is glossed;
- nothing about the *mechanics* depends on the language — a tokenizer, a loss, and an
  attention head behave the same on English text.

---

## 2. Program output is quoted verbatim. Always.

Every ` ```text ` block in the Chinese edition is **real captured output** from a real run.
In the English edition:

- **Copy it byte-for-byte.** It is Chinese. Leave it Chinese.
- **Never translate it. Never regenerate it. Never "clean it up".**
- Where the output carries the argument, add a **one-line English gloss** immediately
  before or after the block. Not inside it.

Example:

```text
  苹果  第  0 号  权重  -0.234
  发布  第  1 号  权重  +1.156
```
*(The table lists each word, its arbitrary index from Chapter 1, and the weight the model
learned for it.)*

Blocks whose content doesn't need translation (pure numbers, loss values, shapes,
attention matrices) need no gloss — use judgement. Gloss when a reader who can't read
Chinese would miss the point; don't gloss when the numbers speak for themselves.

---

## 3. Section headings — use these exact strings

Both `README.en.md` files and the root README use the same eight headings, in the same
order, translated as:

| 中文 | English |
|---|---|
| `## 本章的问题` | `## The problem in this chapter` |
| `## 最简单的尝试` | `## The simplest attempt` |
| `## 实验` | `## Experiment` |
| `## 新机制` | `## The new mechanism` |
| `## Python 实现` | `## Python implementation` |
| `## 它解决了什么` | `## What it solves` |
| `## 它还解决不了什么` | `## What it still can't solve` |
| `## 练习` | `## Exercises` |

The last section of `README.en.md` must keep the Chinese edition's one-line pointer to
`exercises.en.md`.

---

## 4. Terminology — fixed, do not improvise

| 中文 | English | Notes |
|---|---|---|
| 分词 / 把句子拆开 | splitting the sentence apart | **Ch.1 must never say "tokenization" / "tokenizer"** — the Chinese deliberately withholds the name |
| 词表 | vocabulary | |
| token id | token id | |
| 特征 | feature | |
| 系数 | coefficient | Ch.3's hand-written numbers |
| 权重 | weight | the name arrives in Ch.4 |
| 分数 | score | |
| 损失 | loss | |
| 概率 | probability | |
| 梯度 | gradient | |
| 学习率 | learning rate | |
| 隐藏层 | hidden layer | |
| 神经网络 | neural network | |
| 激活函数 | activation function | |
| 非线性 | nonlinearity | |
| 自动求导 | automatic differentiation | |
| 计算图 | computation graph | |
| 反向传播 | backpropagation | |
| 词向量 / embedding | embedding | |
| 上下文 | context | |
| 注意力 | attention | |
| 多头 | multi-head | |
| 位置编码 | positional encoding | |
| 残差连接 | residual connection | |
| 归一化 / LayerNorm | normalization / LayerNorm | |
| Transformer Block | Transformer block | |
| 语言模型 | language model | |
| 因果掩码 | causal mask | |
| 自回归 | autoregressive | |
| 温度 | temperature | sampling |
| top-k | top-k | |
| 指令微调 | instruction tuning | |
| 偏好对齐 | preference alignment | |

Terms that are already English in the Chinese text (`Tensor`, `softmax`, `Adam`, `BPE`,
`GPT`, `ReLU`, `GELU`) stay as they are.

### Concept-ordering discipline carries over

The Chinese edition enforces a strict "a concept may not appear before the chapter that
explains it" rule, and a checker (`check_book.py`) verifies it. **The English edition must
obey the same rule with the same chapter boundaries.** The mapping is in
`WRITING_SPEC.md` §8 — obey it. The most easily-broken ones:

- no "attention" before Chapter 13
- no "Query/Key/Value" before Chapter 14
- no "Transformer" before Chapter 19
- no "residual" before Chapter 17
- no "causal mask" before Chapter 21
- no "fine-tuning" before Chapter 29
- **no "tokenizer"/"tokenization" in Chapter 1**

---

## 5. Voice

Same voice as the Chinese: direct, second person plural ("we"), willing to say
"this looks stupid, bear with us", never breathless, never "simply put".

- Prefer short sentences. English tolerates them better than Chinese does.
- **Translate the reasoning, not the syntax.** If a Chinese sentence's rhetorical shape
  doesn't work in English, rebuild the sentence — but keep the argument identical.
- Do not add claims, caveats, or enthusiasm that aren't in the Chinese.
- Do not soften the book's honesty. When the Chinese says "this experiment failed to show
  what we wanted", the English says exactly that.
- Keep the "we" narrator. Never "the reader", never "one".

---

## 6. File header

Every `README.en.md` and `exercises.en.md` starts with exactly this first line:

```markdown
**English** | [中文](README.md)
```

(for `exercises.en.md` the Chinese link is `[中文](exercises.md)`)

then a blank line, then the content. Do not add a second switcher line anywhere.

---

## 7. Before you report done

- [ ] `README.en.md` and `exercises.en.md` exist for every chapter you own
- [ ] Both start with the switcher line, then a blank line
- [ ] The eight headings are present, exact, in order
- [ ] Every ` ```text ` block is **byte-identical** to the Chinese edition
- [ ] No code file was touched (`git status` if available, or file mtimes)
- [ ] No concept appears earlier than its allowed chapter
- [ ] Every Chinese example word that matters is glossed on first appearance
