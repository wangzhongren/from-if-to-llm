**English** | [中文](exercises.md)

# Chapter 3 exercises

Everything in this chapter is that `COEFFICIENT_TABLE` in `after.py` — 32 numbers.

The exercises below are all "poke these numbers and see what happens". Guess before you change,
run after you change.

---

## 1. Find 芯片's dividing line

Right now 芯片 is `(1.5, 0.0)`. Turn it down to `(0.4, 0.0)` and run it:

```text
正确：9/10
  [错] 苹果芯片很强     科技  0.6 / 食品  1.2   答：食品   真实：科技
```

*(`正确：9/10` — correct 9/10; the sentence that breaks is 苹果芯片很强, technology 0.6 / food
1.2, answered food, truly technology.)*

Now **work it out first, then try it**: in 「苹果芯片很强」, 苹果 gives technology 0.2 and food
0.9, and 很 gives food 0.3. How big does 芯片 have to be for the technology score to beat the
food score?

The answer: 芯片's coefficient has to be **greater than 1.0**.

Try setting it to exactly `1.0` and see what happens. (Hint: `judge` uses `>=`.)

---

## 2. Zero out the whole 苹果 row

苹果 is currently the row `(0.2, 0.9)`. It appears in 7 of the 10 sentences, and it's the only
word that gives a share to both sides at once.

Change it to:

```python
"苹果": (0.0, 0.0),
```

Run it.

**The accuracy is still 10/10.**

A word that appears in 7 sentences, with all its coefficients set to 0, makes no difference at
all. Think it through:

- Did we fill it in for nothing?
- Turn the question around: which other numbers in this table were filled in for nothing? How
  would you know?
- In Chapter 2 we deleted a rule and the accuracy didn't change either. Are these two the same
  thing?

---

## 3. Multiply all the small coefficients by 2

Multiply the coefficients of these four words by 2:

```python
"新":   (1.0, 0.0),
"很":   (0.0, 0.6),
"这个": (0.0, 0.4),
"做成": (1.0, 0.0),
```

Run it. The accuracy is **still 10/10**.

But look at the two scores for 「苹果芯片很强」:

```text
原表：     科技 1.7 / 食品 1.2   赢 0.5
乘 2 后：   科技 1.7 / 食品 1.5   赢 0.2
```

*(Original table: technology 1.7 / food 1.2 — wins by 0.5. After multiplying: technology 1.7 /
food 1.5 — wins by 0.2.)*

The accuracy didn't move a hair, but this sentence is now close to flipping over.

Think it through: **is "accuracy" too crude a measure?** It only sees right or wrong; it can't
see the difference between "barely winning" and "winning comfortably". (In Chapter 5 we'll swap
in a finer ruler.)

---

## 4. Add one sentence

Suppose the corpus gained one more sentence:

```text
苹果很香
```

*(`苹果很香` = "the apple smells great".)*

It gets split into `苹果 | <UNK> | <UNK>` — neither 好 nor 香 is in the vocabulary.

The table's answer for it is "food" (technology 0.2 / food 0.9), and **the answer happens to be
correct**. But the program never saw the character 香; it just got lucky because 苹果 leans food.

To make the program actually see 香, how many places do you have to touch? Make a list:

- Does the vocabulary need a word? (Yes — add 香.)
- Does the coefficient table need a row? (Yes — fill in two numbers for 香.)
- Are the original 10 sentences still all correct? How would you confirm that?

Count it up: **for one new sentence and one unseen word, how many places did you change?** Now
think about 1000 new sentences.

---

## 5. Hand-compute a threshold

A test of whether you really understand where a "score" comes from.

「苹果很甜」 currently scores technology 0.2 / food 2.0.

Question: how far does 甜's coefficient have to fall before this sentence flips to "technology"?

(Don't rush to run the program — set up the expression first: `0.9 + 0.3 + 甜 ? 0.2`. Then run it
to check your answer.)

The answer: **甜 has to fall to -1.0 or lower.**

In other words, 甜 has to **argue against** food before 「苹果很甜」 can be called technology.
Think it through: can a coefficient be negative? What does a negative one mean?

(Yes, it can. It means "when this character appears, this sentence is **not** in this
category".)
(That's also why, when we start moving these numbers around by hand in Chapter 4, we can't just
tack on a restriction like "positive numbers only".)

---

## 6. Challenge: can the program try things for us?

This is how we filled in the numbers in this chapter:

1. Put in an arbitrary set;
2. Run it once and see which sentence is wrong;
3. Change one number;
4. Go back to step 2.

All that changing back and forth was really the same action over and over: **change a bit, see
whether it got better or worse.**

There's an annoying thing about that action — whether the 1st number or the 32nd number gets
changed first, and by how much, is all down to feel.

So: what if the program repeated that action itself?

- How would it know it "got better" or "got worse"? (Right now it only knows how many sentences
  were right.)
- Should it change one number at a time, or all of them?
- How much should it change each time? What happens if it changes too much? What if too little?

You don't have to answer these. Remember the three questions. Chapter 4 starts answering the
first one, and Chapter 5 answers the other two.
