**English** | [中文](教学指南.md)

# Teaching Guide

For whoever runs this as a course. Students *can* read it alone, but it's **not
recommended** — the value of this book is watching things fail, and someone reading solo
will skip ahead to Chapter 22 and copy the code.

---

## 1. How not to teach this

Let's start with what doesn't work.

**❌ Reading the chapters aloud as slides.**
Each chapter's "new mechanism" is unremarkable on its own. Presented that way, this becomes
an ordinary Transformer tutorial — the exact thing this book exists to avoid. All the
leverage is in **the previous chapter's problem**.

**❌ Doing it in one sitting.**
Thirty chapters in one go, and people drift at Chapter 12 and quit at Chapter 17.

**❌ Only running `after.py`.**
`after.py` is correct, clean, and boring. **`before.py` is the main character of this
book.**

---

## 2. Split it into four sessions

90 minutes each. After every session, students should be able to say on their own what the
*next* session has to solve.

### Session 1 — From rules to learning (Chapters 1–6)

| Segment | Content | Time |
|---|---|---|
| Opening | Ask: "A computer gets `苹果发布了新手机` — what does it actually see?" | 5 min |
| Live coding | Ch. 1–2. **Write if/else together, write it until your hands hurt** | 25 min |
| Turn | Ch. 3: turn the `if`s into numbers. "The `if` didn't disappear, it became a number" | 15 min |
| Core | Ch. 4–5: wrong → adjust the weights → one number for "how wrong" | 25 min |
| Close | Ch. 6: gradients. `w = w - lr * grad` | 20 min |

**Session 1 lives or dies on Chapter 2.** Students must genuinely feel the despair of "tens
of thousands of words — do I write tens of thousands of `if`s?". If you cut that short,
everything downstream collapses.

### Session 2 — From neural network to autograd (Chapters 7–9)

| Segment | Content | Time |
|---|---|---|
| Warm-up | Ch. 7, XOR. **Let students try to tune the parameters by hand** — let them fail | 20 min |
| Insight | Ch. 8: demonstrate that 100 stacked linear layers = 1 linear layer | 20 min |
| Centerpiece | Ch. 9: autograd. **Work the `a*b+c` backward pass on the whiteboard** | 40 min |
| Must-cover | `backward()` **accumulates** gradients, same as PyTorch | 10 min |

**Session 2 lives or dies on Chapter 9.** Every gradient bug your students hit in PyTorch
for the next five years traces back to whether this landed. Run the demo where you comment
out `zero_grad()` and watch the loss go to `nan`.

### Session 3 — How attention was forced into existence (Chapters 10–19)

The longest session, and the heart of the book.

| Segment | Content | Time |
|---|---|---|
| Meaning | Ch. 10–11. **Show the similarity table** — meaning becoming spatial structure | 20 min |
| The trap | Ch. 12. **Show that the average context vector's nearest word is `的` ("of")** | 15 min |
| Climax | Ch. 13–14: from "similar" to Q/K/V | 25 min |
| Expand | Ch. 15–16: multiple heads, position | 15 min |
| Assemble | Ch. 17–19: residuals, LayerNorm, the block | 15 min |

**Session 3 lives or dies on Chapter 12's trap.** Students must see with their own eyes
that averaging makes `的` the closest word. Only then does Chapter 13's attention have a
reason to exist.

> ⚠️ Chapter 17's "more layers make it worse" experiment takes a while to run. **Run it
> ahead of time and show the results.**

### Session 4 — It can actually talk (Chapters 20–30)

| Segment | Content | Time |
|---|---|---|
| Closure | Ch. 20: **"we went all the way around and ended up back at classification"** | 15 min |
| Must-cover | Ch. 21: training peeks at the answer. Low loss, garbage output | 20 min |
| Climax | Ch. 22: train the small language model. **Watch it generate, live** | 25 min |
| Generation | Ch. 23–24: temperature / top-k; the probes | 15 min |
| Close | Ch. 25–27: the PyTorch rewrite → the GPT comparison | 10 min |
| Horizon | Ch. 28–30: scale, alignment, back to the first line of code | 5 min |

**Session 4 lives or dies on Chapter 22.** The moment the model first produces coherent
Chinese, the room reacts — that moment is the entire payoff of the course. **Rehearse the
training run beforehand.** Waiting a minute live is fine; crashing on the first attempt
kills the mood.

---

## 3. Three teaching moves

Only three, used over and over:

1. **Run `before.py` first. Ask "what's wrong?"**
   Let them name the problem. If they can't, run it again, or push the data to a more
   extreme case.

2. **Then ask "so how would you fix it?"**
   Let them guess. A right guess is good; a wrong guess is better — a wrong guess proves
   the mechanism wasn't obvious.

3. **Then run `after.py` and say the name.**
   "This thing, from now on, we call it X."

---

## 4. Questions students always ask

| Question | How to answer |
|---|---|
| "Why not just use PyTorch?" | Chapter 25 does. But write it yourself once first, and you'll know what PyTorch is doing for you. |
| "I can't do this math." | You don't need to. Every formula has a matching Python implementation. Running it beats deriving it ten times. |
| "Why does `backward()` accumulate gradients?" | Because one parameter can be used along several paths, and every path's contribution counts. Chapter 9 covers it. |
| "Is the Chapter 22 model useful?" | No. But the difference between it and GPT is scale, not principle — that's Chapters 27 and 28. |
| "What can I do after learning this?" | You'll have judgement that isn't intimidated by terminology. That's worth more than being able to call a library. |

---

## 5. What "done" looks like

After four sessions, have everyone answer three questions:

1. **Why did attention have to appear?** (No Q/K/V allowed — name the problem it solves.)
2. **What problem does the residual connection solve?** (Not "it prevents vanishing
   gradients" — that's memorized.)
3. **What is the essential difference between GPT and the model we trained in Chapter 22?**

Anyone who can answer #3 has actually walked the whole chain.

---

## 6. Presentation notes

- Crank the terminal font size. If nobody can read the output, nothing else matters.
- Put `before.py` and `after.py` side by side, or use two terminal windows.
- **Put these on screen**: Chapter 11's similarity table, Chapter 12's weight table, and
  Chapter 22's generated text.
- Every script is seeded, so what you get matches the book exactly. You can read the output
  straight off the page.
