# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 模型保存恢复脚本
import tensorflow as tf
import os


def create_model(session, Model_class, path, config, logger):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def save_model(sess, model, path, logger, global_steps):
    checkpoint_path = os.path.join(path, "ner.checkpoint")
    model.saver.save(sess, checkpoint_path, global_step=global_steps)
    logger.info("model saved")
