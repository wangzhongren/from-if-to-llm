**English** | [中文](exercises.md)

# Chapter 2 exercises

All the code in this chapter does is line 8 rules up in a queue that runs from top to bottom.
So every exercise here is "nudge the queue and watch it break".

Guess the result before you change the code, then run it.

---

## 1. Move the 很 rule to the very front

In `judge` in `after.py`, cut this block out:

```python
    if "很" in words:
        return "食品"
```

and paste it at the very top of the function (before `if "手机" in words`).

Run it. The accuracy is:

```text
9/10
```

*(9/10.)*

Which sentence is wrong? Why that one, of all of them?

---

## 2. Delete the 很 rule

Now **delete** that block entirely (not move it — delete it) and run it again.

```text
正确：10/10
```

*(`正确` = the number correct: unchanged at 10/10.)*

Not one sentence changed.

So why was that rule ever written? Is it doing anything at all?

Think about how serious this is: **we don't know which of our rules are useful.** People who
write code by patching never know how much of their patching was junk.

---

## 3. For the one word 香, how many places do you have to change?

The current rules can't answer 「苹果好香」 correctly. Let's fix that.

Step one, add a word to `VOCABULARY`:

```python
VOCABULARY = [
    "苹果", "发布", "新", "手机", "芯片", "电脑", "华为", "小米",
    "好吃", "很", "甜", "香蕉", "这个", "真", "做成", "派", "香",
]
```

Run it and see what 「苹果好香」 gets split into now. (Hint: 好 is still an `<UNK>`, because the
vocabulary has 好吃 but not 好.)

Step two, add a rule in `judge` so it answers correctly.

Count it up: for one word, how many places did you change?

Then think: if the corpus contained 10,000 words that aren't covered, how many times would you
have to do this?

---

## 4. Xiaomi's trouble

Run this sentence:

```text
小米很好吃
```

*(`小米很好吃` = "Xiaomi is delicious".)*

The rules answer "technology". Is that wrong? Why?

小米 and 苹果 ran into the same trouble. Say in your own words what that trouble is.

Then think some more: what would 「小米粥」 ("millet porridge") split into? And what would the
rules answer?

---

## 5. Find the rules that are "guesses"

Open `after.py` and sort every rule into two buckets:

- **judgement**: hitting this word basically settles the category;
- **guess**: hitting this word only means we were out of other options.

At least one of them is an obvious guess. Are there any others? For instance, is
`if "很" in words` a judgement or a guess?

One reference standard: **take that rule on its own and use it as the entire program** — hit
means answer "technology", otherwise answer "food" — and see how many it gets right. (Random
guessing is 5/10.) The real results:

```text
只用 "发布" 一条 -> 9/10   （几乎顶得上整个程序）
只用 "苹果" 一条 -> 4/10   （比瞎猜还差）
只用 "很"   一条 -> 3/10   （比瞎猜还差）
```

*(Only the 发布 rule: 9/10, almost as good as the whole program. Only the 苹果 rule: 4/10, worse
than guessing. Only the 很 rule: 3/10, worse than guessing.)*

One rule comes close to full marks, another does worse than a coin flip. And in the code they
look exactly the same: two lines of `if`.

---

## 6. How much does order really matter?

In experiment 2, swapping the positions of two rules dropped the accuracy from 10/10 to 6/10.

Now **reverse** all 8 rules in `judge` (last one first, first one last) and run it.

```text
正确：6/10
```

*(`正确`: 6/10 — just as bad as the swap in experiment 2.)*

Then answer: how many ways are there to arrange these 8 rules?

```text
8 × 7 × 6 × 5 × 4 × 3 × 2 × 1 = 40320
```

*(8! = 40320.)*

Over forty thousand arrangements. If you had to run the corpus once for every arrangement to
check it, how would you go about finding "the best arrangement"?

---

## 7. Challenge: how many words can this set of rules carry?

Where we stand now:

| vocabulary size | number of rules |
|---|---|
| 16 | 8 |

Assume the number of rules is proportional to the vocabulary size. Fill in this table:

| vocabulary size | number of rules |
|---|---|
| 1,000 | ? |
| 50,000 | ? |

Then answer two questions:

1. 50,000 rules — who writes them? And once they're written, who makes sure their order is
   right?
2. If there were a way to make these rules **grow on their own** from the 10 example sentences
   next to them, instead of being hand-written by us — what would you want the result to look
   like?

There's no standard answer to the second question. Carry it with you into Chapter 3.
