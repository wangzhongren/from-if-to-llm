**English** | [中文](exercises.md)

# Chapter 7 Exercises

## 1. Draw those 4 points yourself

Take a sheet of paper, write x1 along the horizontal axis and x2 along the vertical, draw the
four points (0,0), (0,1), (1,0), (1,1), and label each one with its answer. Then take a ruler
and try to place a line that gets them all right.

If you can't, that's exactly right. **Doing it once with your own hands sticks far better than
watching someone else do it once.**

## 2. Make the brute-force search's step smaller

The brute force in `experiment.py` uses a step of 0.1.

Change it to 0.01 and run it again (it will take roughly 10 times as long). Does the best
accuracy change?

Why not? (Hint: multiply w1, w2 and b all by 2 and the line hasn't changed at all. So the
number of genuinely different lines is far smaller than 61³.)

## 3. Switch to a task that "one straight line can separate"

Change the labels to: if **one of the two numbers is 1**, answer 1, otherwise answer 0.

First work out by hand what the four points' answers are, then run the training and see
whether the loss can get below 0.6931.

Then change it to "**only when both are 1** do we answer 1" and run that again.

Can both tasks be separated? How do they differ from this chapter's XOR?
(Hint: write down the four answers for both tasks and compare them with XOR's four answers.)

## 4. Change the middle layer to "shared weights"

Right now `w1` gives each middle unit its own set of weights.

Try letting all the middle units share the same set, `w1[0]` — that is,
`h[j] = w1[0][0] * x[0] + w1[0][1] * x[1] + b1[0]` — so that every `j` computes exactly the
same h.

After a change like that, does the middle layer still mean anything? What does the loss
become?

## 5. The middle layer with only 1 unit, and even with 0

First run with `hidden_size` set to 1, then run with the middle layer's size set to 0.

- Does a two-layer network with 1 unit express more than that one-layer model from Chapter 6?
- What is the loss with 0 units? What is left of this model?

(Write the middle layer's formula out and you can work it out:
`h = w1[0] * x[0] + w1[1] * x[1] + b1`, `分数 = w2 * h + b2` — where `分数` is "score".
Substitute the first line into the second and see what it looks like.)

## 6. A question worth thinking about ahead of time

In this chapter we added a middle layer, and the model's expressive power didn't change at
all.

So then: **what kind of change would let the model express something that is not a straight
line?**

Don't rush off to the next chapter. Think up two possible answers yourself and write them on
paper. Once you've read the next chapter, come back and compare — did you point in the right
direction?
