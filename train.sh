#!/usr/bin/env bash
python train.py \
    --train true \
    --data /media/seven/data/datasets/自然语言处理数据集/中文分词数据集/data \
    --init_checkpoint chinese_L-12_H-768_A-12 \
    --max_seq_len 128 \
    --max_epoch 1 \
    --batch_size 64 \
    --dropout 0.5 \
    --lr 0.001 \
    --optimizer adam \
    --output output/word_cut

