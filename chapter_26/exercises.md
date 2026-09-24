[English](exercises.en.md) | **中文**

# 第 26 章练习

## 1. 转一下 BPE 的旋钮

打开 `after.py`，把 `num_merges` 从 200 改成 50、500、2000，各跑一遍。

记录三件事：

| 合并次数 | 词表大小 | 语料总 token 数 | "苹果不好吃"的切法 |
|---|---|---|---|
| 50 | | | |
| 200 | | | |
| 2000 | | | |

然后回答：

- 合并到多少的时候，"苹果不好吃"变成了 2 个 token？那 2 个 token 是什么？
- 合并次数一直加下去会发生什么？词表会一直变大吗？什么样的片段会被合并出来？

## 2. 手工给词表打补丁，然后承认失败

打开 `before.py`，把 `VOCABULARY` 里加上"不""强""平""板"这几个词，重新跑。

- 生词少了吗？
- 现在换一句"苹果不甜"试试，还缺什么词？
- 你能想出一个"永远不用再补词"的办法吗？（提示：这正是 BPE 在做的事。）

## 3. BPE 的合并顺序为什么不能打乱

在 `after.py` 里把 `encode` 改成**倒着**应用规则：

```python
def encode_wrong(text, merges, unit=None):
    tokens = unit(text) if unit else list(text)
    for pair in reversed(merges):        # 倒过来用，会怎样？
        tokens = merge_pair(tokens, pair)
    return tokens
```

跑一跑，对比 `encode` 和 `encode_wrong` 的结果：

```python
sentence = "苹果不好吃"
print(encode(sentence, merges))
print(encode_wrong(sentence, merges))
```

结果一样吗？为什么一样（或者为什么不一样）？
再试试 `decode(encode_wrong(...)) == 原文` 还成不成立。

## 4. 用真正的字节版跑一遍

GPT-2 的做法和我们的字节版还有一点差别：它会先用一个正则表达式把文本切成"词块"
（英文的单词、空格、标点各自成块），再在每一块内部做 BPE。这样"苹果"和" 苹果"
（前面带空格）会得到不同的 token。

试着在 `train_bpe` 之前加一步预切分：

```python
import re
CHUNK = re.compile(r"\s+|[a-zA-Z]+|[0-9]+|.")

def chunked(text):
    return [c for c in CHUNK.findall(text)]
```

（注意我们的 BPE 需要"一个 char 一个 token"，所以预切分之后还要把每个块拆成字符，
但**合并不能跨块**。想清楚该怎么改代码——这个改动比看上去要麻烦。）

## 5. 数一数中文的 token 成本

GPT-2 的原始词表（5 万 token）是在英文上训练的，中文会被切得很碎。
用我们的字节版 BPE 量一下：

```python
texts = ["苹果很好吃", "床前明月光疑是地上霜", "hello world", "🍎"]
```

- 每种文本，一个字符平均占几个 token？
- 中文和英文，哪个更"贵"？贵多少倍？
- 这对"同一个模型的 API 按 token 收费"意味着什么？

## 6. 让 BPE 也学会"语料里没有的片段"

在 `build_corpus` 里加上几句带"不"字的句子（比如"苹果不甜"、"香蕉不好吃"），
重新训练 BPE，然后看"苹果不好吃"被切成了几个 token。

再多加几句呢？加到多少次，"不好吃"会变成一个 token？

这说明了 BPE 的一条铁律：**它只会合并它在数据里见过的东西。**
分词器看不看得见某个片段，完全取决于语料里出现过多少次。
