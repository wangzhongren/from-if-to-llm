**English** | [中文](exercises.md)

# Chapter 6 Exercises

## 1. Break h and see what happens to the numeric gradient

`numeric_gradient` in `after.py` uses central differences, with `h=1e-3` by default.

Compute the same gradient with `h=1.0`, `h=0.1`, `h=1e-6` and `h=1e-12`, check each against
the formula, and watch how the error changes.

You'll see that both ends are bad: if h is too large, the "ignore the higher-order terms"
step in the derivation stops holding; if h is too small, subtracting two nearly equal
floating-point numbers eats up all the significant digits.

**Conclusion**: the numeric method isn't just slow, it also demands that you hand-pick a
suitable h. That's why we end up deriving the formula by hand.

## 2. Plot the learning rate

Write a loop that takes `lr` from `0.001` to `10`, sampling a value every so often, trains 200
steps for each one, and prints the final loss as a table.

Then answer: on this data, what is the "best" learning rate? Is it the only good choice?

(Hint: because the 10 sentences are separable, a lot of learning rates end up pushing the
loss down very low. The only end that really goes bad is the "too small" one.)

## 3. Deliberately flip the sign in the update formula

Change every `-=` in `train()` to `+=` and run it.

What happens to the loss? Paste the output.

The point of this exercise isn't to get you to fix it, it's to make you understand this:
**if the gradient is computed wrong, training won't raise an error. It will just quietly
learn worse and worse.** That is also the thing the engine we're going to write in Chapter 9
fears most.

## 4. Train only the "food" class

The current `gradient()` produces gradients for both classes at once.

Try updating only the food class's weights and leaving the technology class completely alone,
and see what happens. How low can the loss go? Why can't it go lower?

(Hint: softmax is a normalization — if you go up, I go down. The two classes' weights are a
single whole.)

## 5. Derive the gradient by hand

In `after.py`, that `(p - y)` is handed to you as a finished result.

Try deriving it yourself, starting from these three expressions:

```text
score[c] = 这句话里每个词在 c 这一类上的权重，加起来
p        = softmax(score)
损失      = -log(p[正确答案])
```
*(score[c] = the weights that every word in this sentence has for class c, added up;
p = softmax(score); loss = -log(p[correct answer]))*

First differentiate `p` with respect to `score`, then differentiate `score` with respect to
each word's weight, and multiply the two.

When you're done you'll find that differentiating softmax and cross-entropy separately is a
mess, and yet **put together** the result is shockingly short. That's not a coincidence —
cross-entropy was built to pair with softmax. In Chapter 9 we'll let the machine do the
deriving, but for now you should at least know how a person does it.

## 6. Add a word

Add a word to the vocabulary — for example, change `苹果芯片很强` to `苹果芯片性能很强` and add
`性能` ("performance") to the vocabulary.

Run `after.py` and see what weight `性能` ends up learning, and whether it's positive or
negative.
(Note: only 1 of the 10 sentences contains `性能`, and like `强` ("strong") it appears in that
one sentence only. For a word that shows up so rarely, is the weight it learns trustworthy?)
