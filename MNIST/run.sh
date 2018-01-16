#!/bin/sh
# 上から
# 画像に乗せるノイズのジャンル
# weight_diff(まだ分からん)
# weight_nc(まだわからん)
# 何個画像を生成するか
# 勾配計算の回数
# 閾値


python gen_diff.py 'blackout' \
                   0.1 \
                   10 \
                   10 \
                   10 \
                   10 \
                   0.9

