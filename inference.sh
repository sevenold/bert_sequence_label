#!/usr/bin/env bash
python inference.py \
    --init_checkpoint chinese_L-12_H-768_A-12 \
    --max_seq_len 128 \
    --output output/word_cut