# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 数据处理相关工具脚本
import codecs
import json
import logging
import math
import os
import random
import re
from .conlleval import return_report


def bio_to_json(string, tags_list):
    tags = []
    for _ in tags_list:
        if _ != "O":
            tags.append(_+"-cut")
        else:
            tags.append(_)
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags) - 1:
            tag_next = tags[c_idx + 1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx - 1][2:] or tags[c_idx - 1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
                    entity_name = ''
        iCount += 1
    return item


def bmes_to_json(string, tags):
    """
    中文分词
    """
    item = {"string": string, "entities": []}
    entity_name = ""

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if tag == 'B':
            entity_name = c
        elif tag == 'M' or tag == 'E':
            entity_name += c
        else:
            item['entities'].append(c)
        if tag == 'E':
            item['entities'].append(entity_name)
    return item


class BatchManager(object):

    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i * batch_size): int((i + 1) * batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        """
        把batch整理为一个[5, ]的数组
        :param batch:
        :return:
        """
        strings = []
        segment_ids = []
        chars = []
        mask = []
        targets = []
        for string, seg_ids, char, msk, target in batch:
            strings.append(string)
            segment_ids.append(seg_ids)
            chars.append(char)
            mask.append(msk)
            targets.append(target)
        return [strings, segment_ids, chars, mask, targets]

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, segment_ids, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def convert_single_example(char_line, tag_to_id, max_seq_length, tokenizer, label_line):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为lb
    """
    text_list = char_line.split(' ')
    label_list = label_line.split(' ')

    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)

        tokens.extend(token)
        label_1 = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(tag_to_id["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(tag_to_id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(tag_to_id["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids, label_ids


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.removeHandler(ch)
    logger.removeHandler(fh)
    return logger


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with codecs.open(output_file, "w", 'utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub(r'\d', '0', s)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

