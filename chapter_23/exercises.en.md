**English** | [中文](exercises.md)

# Chapter 23 Exercises

Change the code in `after.py` (or `experiment.py`), run it again, and see whether the output changes. Each exercise gives you a conclusion you can verify yourself.

---

## 1. The two extremes of temperature

`generate` takes a parameter called `temperature`. Try these two values:

```python
generate(model, tokenizer, "床前", 20, temperature=0.01, seed=0)
generate(model, tokenizer, "床前", 20, temperature=100, seed=0)
```

*(The first argument is the prompt 床前, "before my bed"; 20 is how many characters to continue.)*

- Is the first one's output the same as greedy's? Why?
- What does the second one's output look like? Does it still resemble Chinese?
- Now try `temperature=0.0`. `pick_next_token` will raise a `ValueError` outright. Think it through: why doesn't it just quietly substitute "greedy" for the reader? (Hint: when temperature is 0, what is `p ** (1 / 0)`?)

---

## 2. Can temperature be used to implement greedy

In `after.py`, greedy is implemented with `top_k=1`. What if you switch to `temperature=0.0001` (without setting `top_k`)?

- Is the output the same as `top_k=1`?
- Why can it come out "more or less the same"?
- Work it out: if the largest probability is 0.9998 and the runner-up is 0.000119, what is the ratio between them after temperature 0.0001? (Hint: `(0.9998 / 0.000119) ** 10000`.)
- So which of the two implementations is more "robust"? Why is the `top_k=1` branch in `pick_next_token` more reliable?

---

## 3. Find the dividing line between "still readable" and "not readable"

`experiment.py`'s temperature table only goes up to 8.0. Extend it to 12, 16, 20:

```python
for temperature in (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0):
```

Re-run and watch the "longest repeat" column.

- How far down does it fall? When it hits 1, what does that mean?
- From which temperature onward can you no longer hear "Chinese" in what it writes?
- If you treat "longest repeat" as a measure of "it's reciting from memory", does that measure have any flaw?
  (Hint: think about what "longest repeat = 1" means — that's almost 33 random characters.)

---

## 4. Turn top-k and temperature together

In `experiment.py`'s temperature table, add `top_k=3` to every row:

```python
texts = [generate(model, tokenizer, PROMPT, LENGTH,
                  temperature=temperature, top_k=3, seed=seed)
         for seed in range(6)]
```

Re-run.

- With temperature turned to 8, what are "how many kinds out of 6" and "longest repeat"?
- Compare with the temperature 8 above where no top-k was set: which looks more like Chinese? Which has more variety?
- Why doesn't turning the temperature up produce anything too weird once top-k is applied?
- Think about it: why does a real product offer both knobs instead of keeping only one?

---

## 5. Change the opening

Change `PROMPT` to each of these and re-run the temperature table:

```python
PROMPT = "床前明月光疑是"        # 语料里常见的开头
PROMPT = "水水水水水水水水"      # 语料里从来不出现的组合
PROMPT = "白毛浮绿水红掌拨"      # 另一首诗
```

*(A common opening in the corpus; a combination that never appears in the corpus; the other poem.)*

- Which opening has the largest "effective number of candidates"? Why?
- For the opening from outside the corpus, how far does the temperature have to be turned before you can see any change?
- Go back to the five candidate probabilities printed at the top of `experiment.py`. What was that passage about? What does it have to do with the question Chapter 24 asks?

---

## 6. Answer in one sentence

No code needed.

In Chapter 22 we spent an entire chapter assembling a 155,489-parameter model. In Chapter 23 we changed not a single parameter — only "how to pick one of the 33 probabilities".

Now the question: **if you could keep only one knob, would you keep temperature or top-k? Why?**

(Write your answer down first, then go into `experiment.py` and try the two knobs separately.)
