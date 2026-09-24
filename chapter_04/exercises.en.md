**English** | [中文](exercises.md)

# Chapter 4 exercises

The code in this chapter is only 3 functions. So most of the exercises are "change one line and
see whether it can still learn".

What's interesting is that you'll find this chapter's rule **very sturdy** — change it anywhere
and it doesn't get much worse. That's not a good thing. It means the rule is too crude, crude
enough that "however you change it, it still works".

---

## 1. Does how much you change each time really not matter?

Change this in `after.py`:

```python
STEP = 1.0
```

to `0.001` and run it. Then change it to `1000.0` and run it again.

```text
三轮都是一模一样的：1  4  2  0
```

*(All three runs are identical: 1 4 2 0.)*

Think it through: if "how much you change each time" doesn't affect right or wrong at all, then
what did we actually learn during training?

(Hint: look at the table after training. Did the numbers get bigger or smaller?)

---

## 2. Tamper with the three 0s by hand

After training, 5 words in the weight table are 0: 电脑, 华为, 小米, 香蕉 and 很.

Now change them by hand to `[5.0, -5.0]` and run `measure(CORPUS)` again:

```python
WEIGHT_TABLE["电脑"] = [5.0, -5.0]
WEIGHT_TABLE["华为"] = [5.0, -5.0]
WEIGHT_TABLE["小米"] = [5.0, -5.0]
print(measure(CORPUS))
```

```text
还是一样：10/10
```

*(`还是一样：10/10` — still the same: 10/10.)*

Think it through: those three words were changed to completely different values, and the accuracy
didn't move at all.

- So why didn't training adjust them? (Hint: look at that `continue` in `train_one_round`.)
- Is there any difference in the result between a weight that was never adjusted and one that was
  adjusted over and over?

---

## 3. Take the "minus" out

`adjust` has two lines:

```python
WEIGHT_TABLE[word][correct_index] += step
WEIGHT_TABLE[word][wrong_index] -= step
```

Delete the second line (or comment it out), keeping only "add points to the correct side".

Run it.

```text
还能训到 10/10，训练轨迹也一模一样：1  4  2  0
```

*(It still trains to 10/10, and the training trajectory is identical: 1 4 2 0.)*

Why? Think it through: we decide the category by **which of the two scores is larger**. What's
the difference between "give food 1 more point" and "take 1 point away from technology" as far as
"which is larger" is concerned?

---

## 4. Deliberately feed it a wrong answer

Change the label of 「苹果很甜」 in the corpus to "technology":

```python
("苹果很甜", "科技"),
```

(We know it's wrong. The program doesn't.)

Train again, for a full 20 rounds.

```text
它照样训到了 10/10——按照你给的标签算的 10/10。
```

*(It trains to 10/10 anyway — 10/10 measured against the labels you gave it.)*

Think it through: what did the program "learn"? Did it learn "this sentence is technology"? Or
something else?

Then think: if all 10 labels in the corpus were stuck on at random, what would happen?

---

## 5. Hand-compute one round

No running the program. Work it out with pen and paper:

**In the first round**, what happens to the weight table when it reaches the sentence
「苹果很好吃」?

The starting state is all zeros. You need to write down:

- what are this sentence's two scores?
- what does the program answer? Is it right?
- which words are in this sentence? What do each word's two weights become?

When you're done, run `after.py` and check whether the 苹果, 很 and 好吃 rows at the end of round
1 match your calculation.

---

## 6. Challenge: what about three categories?

The corpus right now has only two categories: technology and food. `CATEGORIES` holds two names.

Suppose we had a corpus with three categories (technology, food, sports, say), and we put three
names in `CATEGORIES`.

Can `after.py`'s rule — "wrong means add to the correct one and subtract from the wrong one" — be
used on three categories as it is?

Try writing it out, and you'll run into several problems:

- A sentence was answered "food" but it's really "sports". Which weights should change now?
- The convention "a tie goes to technology" — how do you write that with three categories?
- More troublesome: right now "answered wrong" is a black-and-white state. If the program answers
  "food", and the second most likely is "sports" with technology third — does it care about those
  three situations at all?

(Chapter 5 answers the third question.)
