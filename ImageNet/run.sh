#!/bin/sh
# 上から
# 画像に乗せるノイズのジャンル
# weight_diff(まだ分からん)
# weight_nc(まだわからん)
# 何個画像を生成するか
# 勾配計算の打ち切り回数(Max何回までチャレンジするかの話)
# 閾値


python -u gen_diff.py 'blackout' \
                   0.1 \
                   10 \
                   10 \
                   5 \
                   20 \
                   0.9
                   

