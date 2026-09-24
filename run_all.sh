#!/usr/bin/env bash
#
# 一键验证全书：每一章的 after.py / before.py / experiment.py 都要能跑通，
# 每一章的测试都要全绿。
#
# 用法：
#     ./run_all.sh              # 跑全部
#     ./run_all.sh 13 16        # 只跑第 13 到 16 章
#
set -uo pipefail

cd "$(dirname "$0")"
PY=./.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "找不到虚拟环境 $PY"
    echo "先执行： python3.12 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi

START=${1:-1}
END=${2:-30}

# 先验证引擎本身
echo "=============================================================="
echo " toygrad 引擎自检"
echo "=============================================================="
$PY -m pytest toygrad/tests/ -q
ENGINE_OK=$?
echo

pass=0
fail=0
failed_chapters=""

for n in $(seq "$START" "$END"); do
    ch=$(printf "chapter_%02d" "$n")
    [ -d "$ch" ] || continue

    echo "=============================================================="
    echo " $ch"
    echo "=============================================================="

    ch_ok=1
    for script in before.py after.py experiment.py; do
        [ -f "$ch/$script" ] || continue
        printf '  %-16s ' "$script"
        if out=$($PY "$ch/$script" 2>&1); then
            echo "OK"
        else
            echo "失败"
            echo "$out" | tail -15 | sed 's/^/      /'
            ch_ok=0
        fi
    done

    if [ -d "$ch/tests" ]; then
        printf '  %-16s ' "pytest"
        if out=$($PY -m pytest "$ch/tests/" -q 2>&1); then
            echo "OK"
        else
            echo "失败"
            echo "$out" | tail -15 | sed 's/^/      /'
            ch_ok=0
        fi
    fi

    if [ "$ch_ok" = 1 ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        failed_chapters="$failed_chapters $ch"
    fi
    echo
done

echo "=============================================================="
echo " 总结"
echo "=============================================================="
echo "  通过：$pass 章"
echo "  失败：$fail 章"
[ -n "$failed_chapters" ] && echo "  失败章节：$failed_chapters"
[ "$ENGINE_OK" != 0 ] && echo "  ⚠️  toygrad 引擎自检失败！"

[ "$fail" = 0 ] && [ "$ENGINE_OK" = 0 ] && exit 0 || exit 1
