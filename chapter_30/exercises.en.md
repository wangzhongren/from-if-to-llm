**English** | [中文](exercises.md)

# Chapter 30 Exercises

## 1. Set some hard questions on both exam papers

Expand `TEST_SET` in `after.py` to 30 sentences, deliberately including:

- a sentence with both a tech word and a food word (for instance `苹果手机很甜`, "the Apple
  phone is very sweet");
- a sentence with only new words and none of the old ones (for instance `大米发布新平板`,
  "rice shipped a new tablet");
- a sentence that is grammatical but semantically odd (for instance `香蕉发布新手机`,
  "bananas shipped a new phone");
- a sentence that isn't a sentence at all (for instance `床前明月光`, "moonlight in front
  of my bed").

Then run `experiment.py` and see what the two report cards come to.

- Did the rules side lose points? On which sentences? Why?
- Did the model side lose points? What happens to its probability values ("confidence") on
  the hard questions?
- Which side's failures are more "explainable"?

## 2. Add a third category

Right now there are only two categories, tech / food. Try adding a third, weather, with
sentences in the corpus such as:

```python
WEATHER_SENTENCES = ["今天很冷", "明天很热", "今天有风", "昨天下雨"]
```

*(The four sentences mean "it's cold today", "it will be hot tomorrow", "it's windy
today", "it rained yesterday".)*

Count separately: **how many lines of code does the model side need? How many does the
rules side need?**

- Model side: one more entry in `LABELS`, a few more sentences in the corpus, and `CHARS`
  collects the new characters automatically — how many places did you change?
- Rules side: which new `if`s do you need to write? How many people does it take to think
  up the new word list?
- And at 10 categories? At 100? What does each side turn into?

(This is the modern version of the Chapter 2 question: **the question is not "should we
write rules" but "is the cost of extending the rules linear".**)

## 3. Explain this model with every part you learned in 30 chapters

This stretch of code is the core of `after.py`'s classification:

```python
logprobs = model(torch.tensor([ids + label_ids]))[0].log_softmax(dim=-1)
total = sum(logprobs[len(ids) - 1 + offset, token].item()
            for offset, token in enumerate(label_ids))
```

Match each step here back to the chapter where it first appeared:

| This step | Which chapter it first appeared in |
|---|---|
| `ids = encode(sentence)` turning characters into indices | |
| the table lookup inside `model(...)` | |
| positional encoding | |
| attention | |
| residual | |
| LayerNorm | |
| MLP | |
| `log_softmax` | |
| adding up the log probabilities | |
| comparing two candidates | |

When you have filled it in, you will find that **not one line of this was left untaught by
the book.**

## 4. Write one "thing this book didn't cover"

The last chapter's "what it still can't solve" lists six things: real data, distributed
training, the details of alignment, multimodality, reasoning models, and "what does it
actually understand".

Pick the one you most want to get clear on, and write down these three things:

1. **Where in this book** would it slot in? (For instance, "distributed training" slots in
   after Chapter 28, because that is the first time we wanted to get big.)
2. What foundations does it need that this book **doesn't cover**? (Math, systems, data
   engineering…)
3. How would it change the way you understand "training a model"?

## 5. Add a chapter to this book

Pick a point you think this book skipped over too quickly, write out that chapter's eight
section headings (The problem in this chapter / The simplest attempt / Experiment / The new
mechanism / Python implementation / What it solves / What it still can't solve / Exercises),
and then write only the "The problem in this chapter" and "The new mechanism" sections, 200
words each.

Requirement: **"The problem in this chapter" has to be the question the previous chapter
really left behind.** It can't be something with no lineage, like "and while we're at it,
let's talk about convolutional neural networks".

## 6. The last exercise: break it

Every model in this book is small and well behaved. Try to break it:

- raise the learning rate 100 times (which chapter said what would happen?);
- shuffle the corpus into random characters (where does the loss come to rest? why that
  number?);
- set `block_size` longer than the sentences (what goes wrong with the position table?);
- delete all the training sentences and keep only the validation set (what happens?).

Each time you break it, say clearly: **which chapter's part was not doing its job when it
broke?**
