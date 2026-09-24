**English** | [中文](exercises.md)

# Chapter 29 Exercises

## 1. How expensive is the "alignment tax"

In `after.py`, change the instruction-tuning steps to 30 / 300 / 3000, re-running each time,
and fill in the table:

| Instruction-tuning steps | Answer accuracy | Loss on raw text |
|---|---|---|
| 30 | | |
| 300 | | |
| 3000 | | |

Questions:

- Does more steps always mean higher accuracy?
- From which step does the raw-text loss start getting clearly worse?
- If you could pick only one step count, which would you pick, and why? (Hint: in real
  training this trade-off is called "catastrophic forgetting".)

## 2. Change the preference

In `FACTS`, replace the "verbose answer" with another style you dislike, for instance:

```python
("苹果", "甜", "苹果很甜。", "问：苹果甜吗？答：", "甜。", "这个我不知道。"),
```

*(Reading the tuple: the fact is that `苹果` "apples" are `甜` "sweet"; `苹果很甜。` is the
statement; `问：苹果甜吗？答：` is the prompt; `甜。` is the short answer; and
`这个我不知道。` "I don't know that." is the new verbose answer to be rejected.)*

Re-run stage three of `after.py`. Then answer:

- Can the model learn to dislike this kind of answer? Look at the numbers in the
  probability table.
- What if we dislike both answers? What should the data for this stage look like then?
- Is there a "correct answer" anywhere in this stage's data? Why or why not?

## 3. Run a three-stage ablation yourself

`experiment.py` compares "keep pretraining" against "instruction tuning". Add another
branch: **do preference alignment first, then instruction tuning** (the reverse order).

```python
aligned_first = copy.deepcopy(model)      # starting from the pretrained model
# first run dpo_step on this model ...
# then run sft_step ...
```

Questions:

- With the order reversed, what is the answer accuracy?
- Why does the third step (preference alignment) have to come after the second
  (instruction tuning)? If you reverse them, what is the second stage actually doing?
  (Hint: think about what the data for each stage "looks like".)

## 4. Move the mask position

In `make_sequences`, the mask starts being 1 at `len(prompt) - 1`. Change it to start at 0
(that is, the prompt part counts toward the loss as well), redo the instruction tuning, and
then answer:

- Does the answer accuracy change?
- Did the model learn to "parrot the question"? (Look at whether the generated output
  copies the question back.)
- Why don't real systems do it this way?

## 5. Learn a "scoring" model only

DPO uses one formula to dodge the need for a scoring model. In real systems the more common
approach is **two separate steps**: first train a model that scores answers (a reward
model), then use it to tune the language model.

Try building the simplest version with our model:

```python
# input (question + answer), output one score
scorer = TinyLM(VOCAB_SIZE, dim=64, n_layer=2, n_head=4, block_size=32)
head = torch.nn.Linear(64, 1)     # squeeze the last layer's vector down to one number
```

Train it on the preference data: the good answer should score higher than the bad one (you
can use `-log sigmoid(score difference)` as the loss).

- After training, will it score answers it has never seen?
- Use it to score the model's generations: are the answers it scores highly the ones people
  really prefer?
- Is this "scoring model" the same thing as the language model? (Hint: which layer do they
  differ in?)

## 6. Teach the model to say "I don't know"

Right now the model answers everything bluntly, including the things it doesn't actually
know. Give it some fine-tuning data whose answers are "I don't know":

```python
("苹果", "辣", "苹果不辣。", "问：苹果辣吗？答：", "不辣。", "苹果是辣的。"),
```

*(The fact: `苹果` "apples" are `辣` "spicy" — no, they are `不辣`, "not spicy". The
prompt is `问：苹果辣吗？答：` "Q: are apples spicy? A:", the good answer is `不辣。`
"not spicy.", and the bad answer is `苹果是辣的。` "apples are spicy.")*

(Notice the answer to "are apples spicy" is nowhere in the pretraining corpus — the
pretraining text has no `辣` "spicy" in it at all.)

- How will it answer "are apples spicy"? Guess first, then run it.
- Is it right? Does it "know" that it doesn't know?
- What is this problem (a model that doesn't know what it doesn't know) called in real
  large models?
