# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 超参数设置
import tensorflow as tf


def get_flags():
    flags = tf.flags
    flags.DEFINE_boolean("train", False, "Wither train the model")
    # configurations for the model
    flags.DEFINE_integer("batch_size", 64, "batch size")
    flags.DEFINE_integer("seg_dim", 200, "Embedding size for segmentation, 0 if not used")
    flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
    flags.DEFINE_integer("lstm_dim", 256, "Num of hidden units in LSTM")
    flags.DEFINE_string("tag_schema", "iob", "tagging schema iobes or iob")

    # configurations for training
    flags.DEFINE_float("clip", 5, "Gradient clip")
    flags.DEFINE_float("dropout", 0.5, "Dropout rate")
    flags.DEFINE_float("lr", 0.001, "Initial learning rate")
    flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
    flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")

    flags.DEFINE_integer("max_seq_len", 256, "max sequence length for bert")
    flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
    flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")

    flags.DEFINE_string("output", "output", "Path to save model")
    flags.DEFINE_string("data", "data", "Path for train data")
    flags.DEFINE_string("init_checkpoint", "chinese_L-12_H-768_A-12", "Path to save model")

    FLAGS = tf.flags.FLAGS
    assert FLAGS.clip < 5.1, "gradient clip should't be too much"
    assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
    assert FLAGS.lr > 0, "learning rate must larger than zero"
    assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
    return flags
