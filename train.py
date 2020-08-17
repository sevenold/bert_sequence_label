# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 模型训练和测试模型脚本
import os
import pickle
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from utils.utils import BatchManager
from utils.loader import load_sentences, prepare_dataset, tag_mapping
from model import Model
from utils.utils import get_logger
from tools import create_model, save_model
from utils.utils import print_config, save_config, load_config, test_ner
from config import get_flags
FLAGS = get_flags().FLAGS


# config for the model
def config_model(tag_to_id):
    config = OrderedDict()
    config["num_tags"] = len(tag_to_id)
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config['max_seq_len'] = FLAGS.max_seq_len
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["init_checkpoint"] = FLAGS.init_checkpoint
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.output)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])
    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    tf.io.gfile.mkdir(FLAGS.output)
    log_path = os.path.join(FLAGS.output, 'model.log')
    logger = get_logger(log_path)
    # load data sets
    train_sentences = load_sentences(os.path.join(FLAGS.data, "train.txt"), FLAGS.zeros)
    dev_sentences = load_sentences(os.path.join(FLAGS.data, "dev.txt"), FLAGS.zeros)
    test_sentences = load_sentences(os.path.join(FLAGS.data, "test.txt"), FLAGS.zeros)
    # create maps if not exist
    map_file = os.path.join(FLAGS.output, 'maps.pkl')
    if not os.path.isfile(map_file):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, FLAGS.max_seq_len, tag_to_id
    )
    dev_data = prepare_dataset(
        dev_sentences, FLAGS.max_seq_len, tag_to_id
    )
    test_data = prepare_dataset(
        test_sentences, FLAGS.max_seq_len, tag_to_id
    )
    logger.info("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, FLAGS.batch_size)
    test_manager = BatchManager(test_data, FLAGS.batch_size)
    # make path for store log and model if not exist
    config_file = os.path.join(FLAGS.output, 'config.json')
    if os.path.isfile(config_file):
        config = load_config(config_file)
    else:
        config = config_model(tag_to_id)
        save_config(config, config_file)
    print_config(config, logger)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, os.path.join(FLAGS.output, 'checkpoint'), config, logger)

        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)

                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(iteration, step % steps_per_epoch,
                                                           steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, os.path.join(FLAGS.output, 'checkpoint'), logger, global_steps=step)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def main(_):
    FLAGS.train = True
    train()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)
