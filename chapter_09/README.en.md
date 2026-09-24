**English** | [中文](README.md)

# Chapter 9: Letting the machine compute its own gradients

## The problem in this chapter

At the end of the last chapter, our network looked like this:

```text
输入 -> 中间层（relu）-> 输出 -> softmax -> loss
```
*(input -> middle layer (relu) -> output -> softmax -> loss)*

Its gradient is something we derived with the chain rule, segment by segment, and wrote down
line by line: the output layer's error, the middle layer's error, that switch inside relu —
every place written by hand.

Two layers is still manageable. So the first thing we do in this chapter is stack the network
up to three layers — not because it needs three, but because we want to see clearly what "one
more layer" costs.
With three layers, all this hand-written stuff comes to:

```text
前向 forward：25 行
反向 backward：55 行
一共 80 行，全是人手写的。
```
*(Forward pass: 25 lines. Backward pass: 55 lines. 80 lines in total, every one written by
hand.)*

And every layer you add means:

1. Deriving the chain rule all over again (3 layers is 2 segments of "error times weight",
   4 layers is 3);
2. Rewriting a chunk of code with indices in it;
3. Checking it against Chapter 6's numeric method afterwards — because getting it wrong
   doesn't raise an error, it just quietly learns worse and worse.

We really did check, and this time we got lucky:

```text
        参数           手写的梯度          数值法的梯度           差
  w1[0][1]     -0.00384526     -0.00384526    7.01e-12
  ...
最大的差是 7.01e-12 —— 这次侥幸是对的。
```
*(Columns: parameter; hand-written gradient; numeric method's gradient; difference. The last
line: the largest difference is 7.01e-12 — this time we got away with it, and it was right.)*

The word "lucky" weighs a lot. In a formula a person derived, right and wrong are one index
apart, and when it's wrong nobody tells you.

**This is the problem Chapter 8 left behind. In this chapter we don't change the loss and we
don't change the structure. We change exactly one thing: who computes the gradient.**

## The simplest attempt

Before inventing something new, there are two obvious options. We tried both.

**Option one: keep writing it by hand.**

That's `before.py`. It runs, the loss goes down, accuracy is 100%.
But the table above already said it plainly: three layers is 80 lines, four layers is more,
ninety layers is unthinkable.

**Option two: don't write it at all — use the numeric method.**

Chapter 6's numeric gradient is universal: it recognises only the single number that is the
loss, and it doesn't care how many layers are inside or whether there's a relu. It works with
whatever structure you swap in, and it is absolutely correct.

The price is its speed:

```text
每算一个参数的梯度，要把整个模型重跑两遍。
```
*(For every parameter's gradient, the whole model has to be rerun twice.)*

We have 22 parameters, so that step is 44 runs. More parameters? GPT-3 has 175 billion
parameters, so one step would be 350 billion forward computations.

That road is blocked too.

**One road needs a person to derive it; the other needs brute force. Neither goes anywhere,
so there is only one idea left:**

> Could we get the program to note down, while it computes, how the derivative should be
> taken?

## Experiment

Run `experiment.py`. This is the highlight of the chapter.

### Step 1: Put automatic differentiation and the numeric method face to face

We build a network identical to Chapter 8's using the engine written in this chapter (the
array version, in `toygrad/tensor.py`), run `loss.backward()` once, and then compare it
parameter by parameter against Chapter 6's numeric method:

```text
          参数              自动求导               数值法           差
       w1[5]     -0.1111189403     -0.1111189404    4.80e-11
       b1[1]     -0.1111189403     -0.1111189404    4.80e-11
       b1[2]     -0.0345109118     -0.0345109117    7.71e-11
       b2[0]     -0.0313660121     -0.0313660121    7.01e-12
       ...
（表里只列了 8 个，一共比了 22 个参数）

全部 22 个参数里，最大的差是 8.67e-11。
```
*(Columns: parameter; automatic differentiation; numeric method; difference. Only 8 rows are
listed, and 22 parameters were compared in total. Across all 22 parameters, the largest
difference is 8.67e-11.)*

8.67e-11 — that is the error range of floating point, not "roughly the same". The two sides
are computing the same number.

The numeric method doesn't know there's a relu inside, or a softmax, or how many layers. It
does exactly one thing: nudge the parameter a little and see how the loss changes. That it
agrees means the engine really did get the answer right.

(One reminder: we deliberately started the parameters from random values.
If some middle unit's output happens to be exactly 0, the numeric method is standing right on
relu's kink, and the two sides won't agree — that is not the engine being wrong, it is that
the numeric method has no definition at the kink in the first place. This kind of "looks like
a bug but is actually me standing in the wrong place" situation will come up again.)

### Step 2: Use it to train XOR

```text
  轮数            loss
   0    0.6912194939
  50    0.3598487753
 100    0.0738168432
 150    0.0265746394
 200    0.0149137610
 250    0.0097655151
 300    0.0072041165

准确率：1.00
```
*(Columns: round; loss. Accuracy: 1.00.)*

Chapter 8's network, Chapter 8's task, and not one character of the training process changed.
The only difference is that this time **we didn't write a single line of gradient code**.

### Step 3: What `no_grad()` saves

```text
正常算一遍：这张图上有 11 个节点，requires_grad = True
用 no_grad()：这张图上有 1 个节点，requires_grad = False

两种算法给出的 loss 一模一样：
  0.6912194939
  0.6912194939
```
*(Computing normally: this graph has 11 nodes, requires_grad = True. With no_grad(): this
graph has 1 node, requires_grad = False. The two ways give exactly the same loss.)*

During training we have to keep the books — if we don't, `backward()` has no idea who to send
the gradients to.
But **inference** doesn't need it: we just want a result, and we don't intend to update
anything.

`no_grad()` is that switch. Chapter 23 uses it when the model generates text.

### Step 4: Why one parameter's gradient gets "added"

```text
a 的值           : [ 0.1257 -0.1321  0.6404  0.1049]
一次 backward 之后: [ 0.2515 -0.2642  1.2808  0.2098]   （正好是 2a）
再来一次 backward : [ 0.5029 -0.5284  2.5617  0.4196]   （变成了 4a）

zero_grad() 之后  : [0. 0. 0. 0.]
```
*(The value of `a`; after one backward pass; after another one. The annotations: "exactly 2a",
"it has become 4a", and then after `zero_grad()`.)*

The second time it computes the same loss, and yet the gradient has doubled.

We'll explain this section properly under "the new mechanism", because it is the easiest pit
in this chapter to fall into.

### Step 5: Point this engine back at those 10 sentences from Chapter 6

```text
训练 200 步：loss 0.680240 -> 0.01774767，10 句里对了 10 句。

它学到的权重（科技那一类）：
    苹果  第  0 号  权重  -0.234
    发布  第  1 号  权重  +1.156
    芯片  第  4 号  权重  +1.874
    好吃  第  8 号  权重  -1.277
    甜    第 10 号  权重  -1.131
    ...
```
*(Training for 200 steps: loss 0.680240 -> 0.01774767, 10 out of 10 sentences correct. The
weights it learned (the technology class): each line gives a word, its number, and its weight.
`第 0 号` is "number 0".)*

This engine can do more than play with XOR. Rewrite Chapter 6's sentence classifier with it
and two hundred steps take it back to 100%.

But take a look at the two characters at the far left of every line: **第 0 号, 第 1 号,
第 2 号** ("number 0", "number 1", "number 2"). That is what this chapter leaves for the next
one.

## The new mechanism

### Get the idea straight on the smallest example first

Without looking at any code, look at three numbers:

```text
a = 2, b = 3, c = 4

d = a * b        -> 6
e = d + c        -> 10
```

Now ask: how sensitive is `e` to `a`, to `b`, and to `c` respectively?

After the forward pass we hold exactly one thing, `e = 10`. Now walk back:

```text
e 对自己的梯度 = 1

e = d + c ：
    加法把梯度原样分给两边
    d.grad += 1
    c.grad += 1

d = a * b ：
    乘法把梯度乘上"对方的数值"，再送出去
    a.grad += b * d.grad = 3 * 1 = 3
    b.grad += a * d.grad = 2 * 1 = 2
```
*(The gradient of `e` with respect to itself is 1. For `e = d + c`: addition hands the gradient
out to both sides unchanged, `d.grad += 1`, `c.grad += 1`. For `d = a * b`: multiplication
multiplies the gradient by "the other one's value" before sending it on,
`a.grad += b * d.grad = 3 * 1 = 3`, `b.grad += a * d.grad = 2 * 1 = 2`.)*

Result: `a.grad = 3`, `b.grad = 2`, `c.grad = 1`.

Check it: change `a` from 2 to 2.001 and `e` goes from 10 to 10.003.
**It changed by 0.003, which is exactly `a.grad × 0.001`.**

Now notice something very important:

> Every step of the walk back uses only two things —
> **what this operation is**, and **what its inputs were at the time**.
>
> And during the forward computation, both of those are known.

So we don't need to derive any formula at all. We just have to jot down a note while the
forward computation happens, and the backward pass can walk itself back. That is the entire
idea of this chapter.

### Computation graph: jot down a note at every step

Treat every intermediate result in that expression as a node, draw it, and you have a graph:

```text
     a(2)   b(3)
        \   /
   d = a*b(6)   c(4)
          \    /
      e = d+c(10)
```

This graph is called a **computation graph**.

In code, "jotting down a note" is done like this:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), "*")
    #                              ^^^^^^^^^^^^^ 记下来路

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward     # 记下"该怎么把梯度还给我的输入"
    return out
```
*(The comments: "note where it came from"; "note how to give the gradient back to my inputs".)*

Every node carries a small function on it. That small function is written **during the forward
computation**, and it remembers what those two numbers were at the time (`self.data` and
`other.data`).

So what `backward()` has to do is absurdly simple:

```python
def backward(self):
    self.grad = 1.0
    for node in reversed(拓扑排序的结果):
        node._backward()
```
*(`拓扑排序的结果` is "the result of the topological sort".)*

Starting from the loss, send its bit of gradient back along the graph in reverse.
Each node is responsible for exactly one thing: **take the gradient it received and hand it
out to its inputs according to the rules of its own operator.**

### Topological ordering: why you can't just walk the graph however you like

If all we did was "start from the loss, call a node's `_backward` the moment you reach it, then
recurse into its inputs", there is a class of graph that comes out wrong.

Wrong how? Look at this graph:

```text
a(2)
├──> b = a * 3
│      ├──> c = b * a
│      └──> d = b + c
```

`b` has two downstream nodes: `c` and `d`. Both of those paths add something to `b.grad`.

If we run `b._backward()` before `c` has given its gradient to `b`, then `b` passes on a value
that is too small — and the value it already passed on can never be made up for.

(This mistake can be reproduced in code; see exercise 3 in `exercises.en.md`:
the correct answer is 15, and without the sort it comes out as 18.)

**So we have to sort first: making sure that when any node's turn comes, all of its downstream
nodes have already finished.** That is why topological ordering has to exist.

In code it comes to 8 lines:

```python
def build(node):
    if id(node) in visited:
        return
    visited.add(id(node))
    for child in node._prev:      # 先把所有上游走完
        build(child)
    order.append(node)            # 再把自己放进去

build(self)
for node in reversed(order):      # 倒着走
    node._backward()
```
*(The comments: "walk all the upstream nodes first"; "then put myself in".)*

This is a post-order traversal: walk all the parent nodes first, then record yourself.
The order it produces guarantees that "a parent always comes after its children", so when we
walk it in reverse, every node is guaranteed to have all of its gradients collected by the
time its turn comes.

### Gradients get "added", not "overwritten"

This is the most important passage in this chapter. Read it slowly.

`_backward` writes `+=`, not `=`:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

Why?

Because **a node may have more than one path leading to the loss.**

The simplest example: `x * x`. Here `self` and `other` are the same object, and both paths
lead to the same `x`:

```text
x(3) ──┬──> x * x (9)
       └──┘
```

When `x` changes, both sides of `x * x` change. So `x`'s contribution to the result is the sum
of the two sides: `3 + 3 = 6`, which is exactly `2x`.

If `+=` were written as `=`, the second path would overwrite the first, and the answer would
be 3 — wrong.

In real life this comes up everywhere. Say the same word appears twice in one sentence: that
weight gets used twice, and of course the gradients on the two paths have to be added together.

**What's the cost?**

The cost is: `backward()` "accumulates", it doesn't "reset". Call it twice in a row and the
same gradient gets added twice.

Step 4 of the experiment saw it with its own eyes:

```text
一次 backward 之后: [ 0.2515 -0.2642  1.2808  0.2098]   （正好是 2a）
再来一次 backward : [ 0.5029 -0.5284  2.5617  0.4196]   （变成了 4a）
```
*(After one backward pass it is exactly 2a; after another one it has become 4a.)*

So every round of training **has to clear the previous round's gradients first**:

```python
for p in params:
    p.zero_grad()
loss.backward()
```

Forget to clear them and the gradients pile up round after round, training gets wilder and
wilder, and the program gives you no hint at all.

**PyTorch behaves exactly the same way**, which is why `optimizer.zero_grad()` shows up in
every PyTorch tutorial.

(Look back at Chapter 6's numeric method: every time it computes a gradient it recomputes the
loss from scratch twice and then throws the result away, so it never had this problem.
Automatic differentiation, in exchange for speed, holds on to the gradients — and the price is
that clearing them becomes your responsibility.)

### Give it a name

This set of ideas is called **automatic differentiation** (autograd):

> Every operation jots down in passing "how I came to be" and "how to get back",
> and when a gradient is needed, you start from the result and walk the record backwards.

- Computation graph: the net that gets recorded automatically during the forward pass.
- `backward()`: send the gradient from the result back to the parameters along the computation
  graph.

## Python implementation

`after.py` contains a complete, self-contained scalar engine, about 130 lines, which you can
read in one sitting. It has a single class:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # 叶子节点：没有上游
        self._prev = tuple(_children)      # 我是用谁算出来的
        self._op = _op                     # 用什么算子
```
*(The comments: "leaf node: no upstream"; "who I was computed from"; "with which operator".)*

Plus `+ - * / **`, `exp`, `log`, `relu`, each operator with its own little `_backward`, and
that 8-line `backward()` on top.

Training code that uses it looks like this:

```python
def forward(params, x):
    """三层。中间两层算完都过一遍 relu。

    整个函数里没有一行字是在算梯度 —— 它只是在做乘法和加法。
    """
    layer1 = [...]
    layer2 = [...]
    return scores
```
*(The docstring: "three layers. The two middle ones each pass through relu after computing.
Not one line of this whole function is computing a gradient — it is just doing multiplications
and additions.")*

```python
for _ in range(steps):
    for p in flat:
        p.grad = 0.0          # 梯度是累加的，每一步开始前必须清零
    loss = batch_loss(params, data)
    loss.backward()           # <- 全部梯度，就这一行
    for p in flat:
        p.data -= lr * p.grad
```
*(The comments: "gradients accumulate, so they must be cleared before every step";
"every gradient there is, in this one line".)*

Compare it with `before.py`: the same three-layer network, and over there there are 55 lines
of hand-written `backward`. Here there are none.

**And what about the full array version?**

The scalar version computes one number at a time. A real model computes hundreds of thousands
of numbers at once. So there is an array version at the repository root, in `toygrad/tensor.py`:

- `float` is replaced by `numpy` arrays;
- `self.grad += other.data * out.grad` becomes an element-wise array multiplication;
- a few more operators are added that only arrays need (matrix multiplication, broadcast
  alignment, softmax, table lookup...).

**Not one word of the idea changes.** That is the version `experiment.py` uses, and you have
already seen its results line up with the numeric method.

`after.py` also plants one thing for later: the way that `Value` class is written corresponds
one-to-one with the design of `torch.Tensor` in PyTorch — `data`, `grad`, `backward()`,
`zero_grad()`, even the names are the same.

## What it solves

- A three-layer network, and the gradient code goes from 55 lines to **0 lines**.
- Add a layer, change a layer, swap relu for tanh — **the gradient code doesn't have to change
  at all.**
  Because the gradient isn't written by us, it's computed.
- Automatic differentiation agrees exactly with Chapter 6's numeric method (22 parameters,
  largest difference 8.67e-11).
- The same engine, pointed back at Chapter 6's sentence classification task, gets all 10
  sentences right.
- We no longer have to "fear getting it wrong". Getting it wrong gets caught on the spot by the
  numeric method —
  whereas a hand-written formula, when it's wrong, just quietly learns worse and worse.

Longer term: in Chapter 7 we got "layers", in Chapter 8 we got "bending", and in this chapter
we got "no more hand-derived gradients".

**Only with those three together does "stack as many layers as you want" become possible.**
Every model that comes later — including that small GPT in Chapter 22, including GPT-3 — is
built out of those three things.

### While we're at it: what PyTorch's `loss.backward()` actually does

The `loss.backward()` you write today does exactly these few things:

1. During the forward pass, every tensor with `requires_grad=True` gets recorded into a
   computation graph;
2. `backward()` starts from `loss` and first sets its own gradient to 1;
3. It walks the graph in reverse; each operator works out its share of the gradient and
   **accumulates** it into its inputs' `.grad`;
4. When it's done, every parameter's `.grad` is sitting there, waiting for `optimizer.step()`
   to read it.

The only difference from the `Value` you wrote in `after.py` is that PyTorch's operators are
array versions, they run on a GPU, and there are more than a thousand of them.
**The principle is identical.**

So: why must `optimizer.zero_grad()` be written? Because step 3 accumulates. That is what this
chapter has been about from beginning to end.

## What it still can't solve

In step 5 of the experiment we pointed this engine back at those 10 sentences and got all 10
right.
But the output is hiding the real problem of this chapter:

```text
它学到的权重（科技那一类）：
    苹果  第  0 号  权重  -0.234
    发布  第  1 号  权重  +1.156
     新  第  2 号  权重  +1.039
    芯片  第  4 号  权重  +1.874
    好吃  第  8 号  权重  -1.277
    甜    第 10 号  权重  -1.131
```
*(The weights it learned (the technology class); each line gives the word, its number, and the
weight.)*

Number 0, number 1, number 2 — those numbers are ones we assigned casually back in Chapter 1.

What we feed the model is a 16-dimensional count vector: position 0 is how many times `苹果`
appeared, position 1 is how many times `发布` appeared, and so on. The model has never seen the
characters `苹果`; all it sees is "the number sitting at position 0".

So here is the question:

- `苹果` is number 0, `香蕉` is number 11. Is there any relationship between those two numbers?
- No. Between 0 and 11 there is no meaning of "fruit", and no meaning of "something you can
  eat".
- If we had ordered the vocabulary differently back in Chapter 1, so that `苹果` became number
  7, would the model learn anything different? **Not in the slightest.**

To the model, the difference between `苹果` and `香蕉` is the same kind of thing as the
difference between "position number 0" and "position number 11" —
**they are just numbers, and the numbers themselves carry no meaning.**

We now have a very usable engine that can compute the gradient of anything.
But the gradients it computes are gradients with respect to *numbers*. Numbers carry no
meaning — so what is it actually learning?

**What the next chapter has to ask is: what else can a word be represented by, besides a
number?**

## Exercises

See `exercises.en.md`. Three of the most important ones:

1. Add a `tanh` or a `maximum` to `Value`, then check the `_backward` you just wrote with the
   numeric method.
2. Change every `+=` in `_backward` to `=` and run `(x * x).backward()`.
   What does `x.grad` become? What should it be?
3. Delete the topological sort and run these three lines:

   ```python
   a = Value(2.0); b = a * 3.0; c = b * a; d = b + c
   ```

   `a.grad` should be 15; without the sort it comes out as 18. Where does the extra 3 come
   from?
