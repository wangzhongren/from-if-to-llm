**English** | [中文](README.md)

# Chapter 20: Bringing the classifier back

## The problem in this chapter

Last chapter we assembled the Transformer block. What it can still do is: turn one string of vectors into another string of vectors.

But what we want is a machine that can write.

One character at a time. Give it `床前` ("before my bed") and it continues with `明` ("bright"); give it `床前明` and it continues with `月` ("moon").

Let's not rush to pile more things on top. Let's go back to the simplest place and take a look first, because there's a fact here that might make you stop for a moment.

What we did in Chapter 2 was **classification**:

```text
"苹果发布新手机"  ->  科技
"苹果很好吃"      ->  食品
```

*(It reads "Apple shipped a new phone" and answers 科技, "tech"; it reads "apples are delicious" and answers 食品, "food".)*

What we're doing in Chapter 20 is:

```text
"床前明"          ->  ?
```

*(The input is "moonlight before my bed, bri…" and the answer is a character.)*

The answer is 月, "moon". But isn't 月 a class label? — **It is.** The only difference is that our candidates went from 2 (tech / food) to 33 (one per character).

If the vocabulary were tens of thousands of words, the output of this step would look like this:

```text
"今天天气很" ->   好 0.52
                  冷 0.21
                  热 0.12
                  苹果 0.0001
```

*(Same shape: "the weather today is very…" → 好 "good" 0.52, 冷 "cold" 0.21, 热 "hot" 0.12, 苹果 "apple" 0.0001.)*

See it? **The last step GPT does is the thing from Chapter 2.** Score a pile of candidates, pick one.

We went around for 18 chapters, from if/else all the way to attention, and ended up back at `score = w · x` plus a softmax.

In this chapter we implement this "new classifier". No attention yet — just the simplest model — and see whether it works, and where it doesn't.

## The simplest attempt

First, bring over the classifier from Chapter 3 unchanged. It looks like this:

```python
score = W[当前字符]
```

*(`W[the current character]` — one set of scores per kind of input.)*

One set of scores per kind of input. There are 33 kinds of input (33 characters), and 33 classes, so it's a single 33×33 table.

Run `before.py`:

```text
============================================================
语料
============================================================
  床前明月光疑是地上霜举头望明月低头思故乡
  鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波
  重复 8 遍，一共 304 个字符，33 个不同的字符
  词表：上举乡低光前向地天头床思拨掌故明是曲月望歌毛水波浮清疑白红绿霜项鹅

  切成 296 个 (输入字符, 下一个字符) 对

============================================================
训练过程
============================================================
  第 100 步   loss 2.3730
  第 200 步   loss 1.4784
  第 300 步   loss 0.9083
  第 400 步   loss 0.5582
  第 500 步   loss 0.4090
  第 600 步   loss 0.3921

  训练集上的 loss     0.3646
  训练集上的准确率    92.23%
```

*(The corpus is Li Bai's *Quiet Night Thoughts* — "moonlight before my bed, I took it for frost on the ground; I raise my head and gaze at the bright moon, I lower my head and think of home" — and Luo Binwang's *Ode to the Goose* — "goose, goose, goose, neck curved, singing to the sky; white feathers float on green water, red feet paddle through clear waves" — repeated 8 times, 304 characters, 33 distinct characters. The vocabulary line lists those 33 characters; the training lines are "step 100…600, loss…". Nothing above the training block is a result: it is the corpus, its size, and the vocabulary. The two numbers that matter: loss 0.3646, accuracy 92.23%.)*

92.23%. That figure is neither high nor low; it lands exactly where it's uncomfortable.

## Experiment

Where does it get things wrong? `before.py` lists every mistake it makes:

```text
============================================================
它答错了哪些地方
============================================================
  语料里的 '头' 后面跟着 '思'，它却猜了 '望'
  语料里的 '鹅' 后面跟着 '曲'，它却猜了 '鹅'
  语料里的 '月' 后面跟着 '光'，它却猜了 '低'
```

*(The header is "where it got things wrong". Each line reads "in the corpus, 头 is followed by 思, but it guessed 望" — 头 "head", 望 "gaze", 思 "think of / miss"; then "鹅 is followed by 曲, but it guessed 鹅" — 鹅 "goose", 曲 "curved"; then "月 is followed by 光, but it guessed 低" — 月 "moon", 光 "light", 低 "low, lower".)*

Only three kinds of mistake — but it made them 23 times (the corpus repeats 8 times).

The reason is plain from the corpus:

```text
...举头望明月...      '头' 后面是 '望'
...低头思故乡...      '头' 后面是 '思'
```

*(…举头望明月 "I raise my head and gaze at the bright moon" — after 头 comes 望. …低头思故乡 "I lower my head and think of home" — after 头 comes 思.)*

Two 头's, followed by different characters. But this classifier **only sees the single character 头**. It has no way to know which 头 this one is, so it bets the same answer every time: right half the time, wrong half the time.

Same story for 月:

```text
...床前明月光...      '月' 后面是 '光'
...举头望明月...      '月' 后面是 '低'
```

*(…床前明月光, "before my bed, bright moonlight" — after 月 comes 光; …举头望明月, "I raise my head and gaze at the bright moon" — after 月 comes 低.)*

So the problem is clear: **it doesn't know how far into the sentence we are.** It only knows "what character most often follows this character".

In Chapter 12 we said "the meaning of a word depends on its context". What we're saying now is the other face of the same thing: **which character comes next also depends on the context.**

## The new mechanism

We change two things.

First, **stop using the index**. In Chapter 10 we said the index means nothing — 头 is number 10, 低 is number 2, and there is no relation at all between that 10 and that 2. Give each character a set of numbers (an embedding table), and let that set of numbers be learned:

```python
vectors = embedding(table, char_ids)     # (B, 33) -> (B, 33, D)
```

Second, **stop looking at one character only — look at the whole stretch in front of it.**

There's a problem here: how many characters in front? If we say "look at 8", the input is 8 sets of numbers. How do we turn those into something we can feed to a linear layer? We did the same thing back in Chapter 7 — **concatenate them**:

```python
flat = vectors.reshape(B, WINDOW * DIM)  # 8 个 16 维的向量 -> 一个 128 维的向量
```

*(8 vectors of 16 dimensions each → one 128-dimensional vector.)*

And then it's the linear layer from Chapter 3:

```python
logits = flat @ weight + bias             # (B, 128) -> (B, 33)
probabilities = logits.softmax(axis=-1)   # 33 个候选，各自一个概率
loss = cross_entropy(logits, answer)      # 第 5 章那个"错得有多离谱"
```

*(33 candidates, each with its own probability; then the Chapter 5 measure of "how badly wrong".)*

That's the entire model. No attention, no MLP, no residual, no LayerNorm.

One more thing while we're here. The first step of the flow above is "turn a stretch of text into a string of ids":

```python
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}
CORPUS_IDS = np.array([CHAR_TO_ID[char] for char in CORPUS])
```

This is exactly what we did in Chapter 1 — split a sentence into tokens, then give each token a number. Back then we didn't name it. Now we can: **this thing that turns text into ids is called a tokenizer.** The step in the other direction is called decode.

The tokenizer we use in this chapter is the dumbest kind: one character is one token, and the vocabulary is every character that appears. A real tokenizer is far more involved (we'll build one ourselves in Chapter 26), but the three things it has to do were settled today: **encode, decode, and know how big its own vocabulary is.**

**Now we can give it a name.**

A model that takes a stretch of text and outputs a probability distribution over the next character, we call a **language model**. What it does is called **next-token prediction**.

You've heard the term before. But what you should remember today isn't the term — it's the very plain thing behind it:

> A language model = a classifier. The number of classes = the size of the vocabulary.

Chapter 2 had 2 classes, Chapter 20 has 33, GPT has over a hundred thousand. The difference is only in the count, not in the nature.

## Python implementation

The model in `after.py` has only three parameter tensors:

```python
class NextCharModel:
    def __init__(self, vocab_size, dim=DIM, window=WINDOW, seed=SEED):
        self.table = randn(vocab_size, dim, scale=0.3, requires_grad=True, seed=seed)
        self.weight = randn(window * dim, vocab_size, scale=0.3, requires_grad=True, seed=seed + 1)
        self.bias = zeros(vocab_size, requires_grad=True)

    def __call__(self, windows):
        vectors = embedding(self.table, windows)          # (B, 窗口, 维度)
        flat = vectors.reshape(windows.shape[0], self.window * self.dim)
        return flat @ self.weight + self.bias
```

*(In the first line of `__call__`, the comment reads "(B, window, dimension)".)*

The way the training data is cut — watch this closely in this chapter:

```python
for position in range(window, len(ids)):
    inputs.append(ids[position - window:position])   # 窗口是 [position-window, position)
    targets.append(ids[position])                    # 答案是 position 上的那个字符
```

*(The comments read "the window is [position-window, position)" and "the answer is the character at position".)*

**The position the answer sits at is not inside the window.** Right now this sentence looks like it goes without saying. Next chapter it becomes the main character.

Run it:

```text
============================================================
语料
============================================================
  床前明月光疑是地上霜举头望明月低头思故乡
  鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波
  重复 8 遍，一共 304 个字符，33 个不同的字符

  切成 296 个 (前 8 个字符, 下一个字符) 对
  模型参数一共 4785 个（其中 4224 个在最后那个线性层里）

============================================================
训练过程
============================================================
  第 100 步   loss 0.0241
  第 200 步   loss 0.0096
  第 300 步   loss 0.0060
  第 400 步   loss 0.0042
  第 500 步   loss 0.0033
  第 600 步   loss 0.0026

  训练集上的 loss     0.0026
  训练集上的准确率    100.00%
  对比：before.py 里那个只看一个字符的分类器是 92.23%
```

*(Same corpus, now cut into 296 "first 8 characters → next character" pairs; the model has 4,785 parameters, 4,224 of them in that last linear layer. Loss reaches 0.0026 and training accuracy 100.00%, against 92.23% for the one-character classifier in `before.py` — that is what the last line says.)*

Then we ask it "which character comes next, and how likely is each one":

```text
  如果词表是几万个词，这一步的输出长这样：
    '今天天气很' -> 好 0.52   冷 0.21   热 0.12   苹果 0.0001
  我们的词表只有 33 个字符，所以长这样：

  '床前明月光疑是地' 后面，下一个字符的候选：
    1. '上'  0.9973  #######################################
    2. '向'  0.0004
    3. '霜'  0.0004
    4. '乡'  0.0003
    5. '床'  0.0003

  '白毛浮绿水红掌拨' 后面，下一个字符的候选：
    1. '清'  0.9975  #######################################
    2. '望'  0.0003
    3. '是'  0.0003
    4. '白'  0.0002
    5. '明'  0.0002
```

*(The first two lines restate the tens-of-thousands-of-words example for comparison. Below them are two "candidates for the next character" lists. After '床前明月光疑是地' it picks 上 ("on / onto", finishing 地上 "on the ground") with probability 0.9973; after '白毛浮绿水红掌拨' it picks 清 ("clear") with probability 0.9975. The bars are just a picture of the probability.)*

Now the key comparison. The **last character of both windows below is 头**:

```text
  '光疑是地上霜举头' 后面，下一个字符的候选：
    1. '望'  0.9979  #######################################
    2. '是'  0.0003
    3. '拨'  0.0003
    4. '项'  0.0003
    5. '光'  0.0002

  '霜举头望明月低头' 后面，下一个字符的候选：
    1. '思'  0.9978  #######################################
    2. '头'  0.0004
    3. '鹅'  0.0002
    4. '月'  0.0002
    5. '是'  0.0002
```

*(Both windows end in 头 "head". The first one ends …举头 "raise head", and the model answers 望 "gaze" — it can see the 举 "raise" just before. The second one ends …低头 "lower head", and the model answers 思 "think of" — it can see the 低 "lower".)*

Same 头: if the character before it is 举, it continues with 望; if the character before it is 低, it continues with 思. The classifier in `before.py` is forever wrong in this cell — it only has one character's worth of information.

## What it solves

Accuracy goes from 92.08% to 100%, loss from 0.3619 down to 0.0026. All three mistakes are gone.

But more important: we have finally **named the goal**. For the previous 19 chapters we built a pile of parts: embedding, attention, MLP, residuals, LayerNorm. Each one's motivation was clear, but what they were being assembled *for* was never said. Now it's said:

> All of these parts exist so that one classifier can classify more accurately.

And this classifier has something special about it: **its classes are the vocabulary itself.** That is, the 33 probabilities the model outputs are a complete guess at "which character should be written next".

One thing to be careful about, though: **our corpus is only 304 characters long, and it has seen all of them.** That 100% says it memorized the corpus, not that it learned Chinese. Chapter 24 will come back to this specifically.

## What it still can't solve

`after.py` ends with a note:

```text
  我们这个模型的输入是语料的 [t-8, t)，答案是 t 位置上的字符。
  答案不在输入里。这 600 步训练，它一直是诚实的。

  但是回头看看我们的训练数据是怎么切的：
    输入：ids[t-8 : t]     答案：ids[t]
  如果换成「一次喂进整段语料、让每个位置都输出下一个字符」，
  那就变成：输入 ids[t : t+T]，答案 ids[t+1 : t+T+1]。
  对窗口里的第 i 个位置来说，它要预测的是 ids[t+i+1]，
  而这个字符，正好是窗口里第 i+1 个位置上的**输入**。
```

*(For this model the input is the corpus slice [t-8, t) and the answer is the character at position t — the answer is not in the input, so across these 600 steps it was honest. Then it asks what happens if we instead feed the whole corpus in at once and make every position output its next character: the input becomes ids[t : t+T] and the answers ids[t+1 : t+T+1], so for the i-th position in the window the target is ids[t+i+1] — and that character is exactly the **input** sitting at position i+1 of the window.)*

Here's the problem:

**Next chapter we're putting attention back. And what attention does is precisely "let every position see every position". Position t will see position t+1 — which is the very character it has to predict.**

This is not an abstract worry. `experiment.py` lays it out. Three groups of models, identical structure, same window, same number of training steps — only the contents of the input slots differ:

```text
  A 组（最后一格是当前字符）
      loss 0.0870      准确率 94.70%      抄答案 3.0%

  B 组（最后一格是答案）
      loss 0.0075      准确率 100.00%      抄答案 72.7%

  C 组（最后一格是答案，上下文随机）
      loss 0.0137      准确率 100.00%      抄答案 100.0%
```

*(Group A: the last slot holds the current character — loss 0.0870, accuracy 94.70%, copies the answer 3.0% of the time. Group B: the last slot holds the answer — loss 0.0075, accuracy 100.00%, copies 72.7%. Group C: the last slot holds the answer and the context is random — loss 0.0137, accuracy 100.00%, copies 100.0%. "抄答案" is literally "copying the answer".)*

"Copying the answer" means: the output is exactly the input's last slot. The probe results make it clearest:

```text
  C 组（最后一格是答案，上下文随机）
    上->上  举->举  乡->乡  低->低  光->光  前->前  向->向  地->地  天->天  头->头  床->床  思->思
    拨->拨  掌->掌  故->故  明->明  是->是  曲->曲  月->月  望->望  歌->歌  毛->毛  水->水  波->波
    浮->浮  清->清  疑->疑  白->白  红->红  绿->绿  霜->霜  项->项  鹅->鹅
```

*(Each pair reads "input character → output character", and every one of the 33 pairs is a character mapped to itself: feed it 上, it says 上; feed it 举, it says 举; and so on down the list.)*

33 probes, 33 times it spits the character straight back out.

Group C is the most glaring: its context is **random characters**, and a model doing anything sensible should learn nothing at all from that input. Yet its accuracy is still 100%.

**As long as the answer appears somewhere in the input, the model will find it. And it will push the loss down, so that you believe the model got better.**

So this is the situation we face next chapter:

Put attention back, and the model can see the whole sequence. But our training data is "input `ids[t:t+T]`, answer `ids[t+1:t+T+1]`" — the answer at position t is the input at position t+1. So every position can see its own answer.

It will learn astonishingly fast, and the loss will drop lower than anything else.

**How do we block it?**

## Exercises

See `exercises.en.md`. Here are the 4 most important ones:

1. Change `WINDOW` from 8 to 4, to 2, to 1, and watch where accuracy lands. At which value does that 头 mistake come back?
2. Change `DIM` from 16 to 2. Can `after.py` still reach 100%? This tells you the final linear layer is doing something very "dead" — try to describe in one sentence what it's doing.
3. In `experiment.py`, change Group B's second slot from "the answer" to "the character **after** the answer". What does the copy-the-answer rate become? Why?
4. `after.py`'s `parameter_count()` says 4,224 of the 4,785 parameters sit in that last linear layer. If you change the window from 8 to 32, what does the parameter count become? And if you want it to look at 1,000 characters?
