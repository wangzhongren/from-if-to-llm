"""toygrad —— 我们在第 9 章亲手造出来的自动求导引擎。

它只做一件事：

    记住每一次运算是怎么算出来的，
    然后沿着原路把梯度倒着送回去。

PyTorch 的 ``torch.Tensor`` 帮我们做的事，就是这个文件里做的事，
只是它更快、算子更多、还能跑在 GPU 上。
"""

from .tensor import (
    Tensor,
    cross_entropy,
    embedding,
    layer_norm,
    masked_fill,
    no_grad,
    randn,
    zeros,
)
from .optim import SGD, Adam

__all__ = [
    "Tensor",
    "cross_entropy",
    "embedding",
    "layer_norm",
    "masked_fill",
    "no_grad",
    "randn",
    "zeros",
    "SGD",
    "Adam",
]
