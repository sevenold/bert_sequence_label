# !/user/bin/env python
# -*- encoding: utf-8 -*-
# @Author  : Seven
# @Function: 序列标注客户端
import json
import pickle
import requests
import numpy as np
import tokenization


def convert_single_example(char_line, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为lb
    """
    text_list = char_line.split(' ')

    tokens = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)

        tokens.extend(token)

    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids


def input_from_line(line, max_seq_length, tag_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    string = [w[0].strip() for w in line]
    char_line = ' '.join(string)  # 使用空格把汉字拼起来
    text = tokenization.convert_to_unicode(char_line)

    tags = ['[CLS]' for _ in string]

    labels = ' '.join(tags)  # 使用空格把标签拼起来
    labels = tokenization.convert_to_unicode(labels)
    tokenizer = tokenization.FullTokenizer(vocab_file='./vocab/vocab.txt',
                                           do_lower_case=True)
    ids, mask, segment_ids = convert_single_example(char_line=text,
                                                    max_seq_length=max_seq_length,
                                                    tokenizer=tokenizer)
    segment_ids = np.reshape(segment_ids, (1, max_seq_length))
    ids = np.reshape(ids, (1, max_seq_length))
    mask = np.reshape(mask, (1, max_seq_length))
    return [string, segment_ids, ids, mask]


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


def decode(logits, lengths, matrix, tag_to_id):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * len(tag_to_id) + [0]])
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)
        paths.append(path[1:])
    return paths


def up_data(features, tag_to_id):
    """
    模型预测数据并返回预测结果
    :param features:
    :return:
    """
    string, segment_ids, chars, mask = features
    payload = {
        "instances": [{'input_ids': chars.tolist()[0],
                       "input_mask": mask.tolist()[0],
                       "segment_ids": segment_ids.tolist()[0],
                       "dropout": 1.0}]
    }
    r = requests.post('http://localhost:8501/v1/models/docker_test:predict', json=payload)
    # print(r.content.decode('utf-8'))
    pred_text = json.loads(r.content.decode('utf-8'))['predictions']
    scores = np.array(pred_text)
    length = len(string) + 1
    trans = np.load("vocab/trans.npy")
    batch_paths = decode(scores, [length], trans, tag_to_id)
    return batch_paths


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


def get_result(msg: str):
    with open("vocab/maps.pkl", "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)

    data = input_from_line(msg, max_seq_length=128, tag_to_id=tag_to_id)
    result = up_data(data, tag_to_id)
    tags = [id_to_tag[idx] for idx in result[0]]
    return bio_to_json(data[0], tags[1:-1])


if __name__ == '__main__':
    text = "中国你好成都。"
    res = get_result(text)
    print(res)
