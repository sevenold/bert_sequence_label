# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 模型推理脚本
import os
import pickle
import tensorflow as tf
from model import Model
from tools import create_model
from utils.loader import input_from_line
from train import FLAGS, load_config
from utils.utils import get_logger


def main(_):
    config_file = os.path.join(FLAGS.output, 'config.json')
    log_file = os.path.join(FLAGS.output, 'model.log')

    config = load_config(config_file)
    config['init_checkpoint'] = FLAGS.init_checkpoint
    logger = get_logger(log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    map_file = os.path.join(FLAGS.output, 'maps.pkl')
    with open(map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, os.path.join(FLAGS.output, 'checkpoint'), config, logger)
        text = "中国你好成都"
        result = model.evaluate_line(sess, input_from_line(text, FLAGS.max_seq_len, tag_to_id), id_to_tag, export=True)
        print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)

