"""仓库根目录的 pytest 配置。

解决一个坑：每一章都有 `after.py` / `before.py` / `experiment.py`，
名字全都一样。各章测试里写的是 `from after import ...`，
一旦在同一个进程里连续跑多个章节，第二次 import 会直接命中
sys.modules 里上一章留下的那个 `after`，于是拿到的是**别的章**的代码。

单章跑（`pytest chapter_13/tests/`）不会触发，多章一起跑或
在仓库根目录跑 `pytest` 就会。这个文件在每次收集一个新模块之前
把上一章留下的同名模块清掉，让每一章都拿到自己的那一份。
"""

import sys

# 各章之间会同名冲突的模块
CHAPTER_LOCAL_MODULES = ("after", "before", "experiment")


def _purge():
    for name in list(sys.modules):
        # 只清顶层模块，别碰 after.xxx 这种包内子模块
        if name in CHAPTER_LOCAL_MODULES:
            del sys.modules[name]


def pytest_collectstart(collector):
    _purge()


def pytest_pycollect_makemodule(module_path, parent):
    # 双保险：pytest 不同版本调用这两个钩子的时机略有差别
    _purge()
    return None
