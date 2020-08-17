## 基于BERT-BLSTM-CRF 序列标注模型
> 本项目基于谷歌官方的BERT：https://github.com/google-research/bert  
> 对BERT进行迁移学习，扩展BLSTM-CRF使模型支持序列标注任务  
> 1. 中文分词
> 2. 词性标注
> 3. 命名实体识别
> 4. 语义角色标注

### 环境配置
- #### miniconda安装

    ```shell
    $ wget -c http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
    $ chmod +x Miniconda-latest-Linux-x86_64.sh
    $ ./Miniconda-latest-Linux-x86_64.sh
    ```

    > 关闭终端，重新打开终端

- #### 项目运行环境配置

    ```shell
    $ conda create -n BERT python=3.6 cudatoolkit==10.0.130 cudnn==7.6.4
    $ source activate BERT
    $ pip install -r requirements.txt
    ```

### 数据准备
> 1. 数据按照如下格式进行整理
> 2. 句子间用换行分开 
> 3. 分割训练数据、测试数据和开发数据（一般为7:2:1）
#### 中文分词数据
```text
义	B
诊	I
当	B
天	I
共	B
有	I
3	B
0	I
0	I
多	O
名	O
群	B
众	I
接	B
受	I
义	B
诊	I
，	O
并	O
直	B
接	I
受	B
益	I
。	O
```

#### 词性标注数据
```text
一  B-m
、  B-w
给  B-v
予  I-v
辽  B-ns
宁  I-ns
队  B-n
和  B-c
山  B-ns
东  I-ns
队  B-n
严  B-a
重  I-a
警  B-v
告  I-v
。   B-w
```

#### 命名实体识别数据
```text
千 O
鹤 O
金 O
是 O
在 O
国 O
家 O
推 O
动 O
下 O
。 O

有 O
大 B-NAME
象 I-NAME
联 I-NAME
盟 I-NAME
发 O
行 O
的 O
我 O
们 O
中 O
国 O
首 O
个 O
承 O
认 O
的 O
数 B-DESC
字 I-DESC
货 I-DESC
币 I-DESC
。 O
```

#### 语义角色标注数据
```text
奥 B-A0
巴 I-A0
马 I-A0
昨 B-TMP
晚 I-TMP
在 O
白 B-LOC
宫 I-LOC
发 O
表 O
了 O
演 B-A1
说 I-A1
```

### 训练模型
#### 下载预训练模型
[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
> 解压到根目录
#### 参数设置
```bash
#!/usr/bin/env bash
python train.py \
    --train true \
    --data data \  # 训练数据保存路径 
    --init_checkpoint chinese_L-12_H-768_A-12 \  # 预训练模型保存路径
    --max_seq_len 128 \  # 句子最长长度
    --max_epoch 1 \  # 模型训练轮数
    --batch_size 64 \  # 模型迭代一次训练的句子数
    --dropout 0.5 \  # 防止过拟合的神经元随机失活率
    --lr 0.001 \  # 学习率
    --optimizer adam \  # 模型优化器
    --output output  # 训练模型及日志保存路径 
```
#### 开始训练
```bash
./train.sh
```
#### 训练日志
```text
I0117 11:15:14.439068 139934521526016 train.py:38] evaluate:dev
I0117 11:15:15.194412 139934521526016 train.py:42] processed 6788 tokens with 3818 phrases; found: 3798 phrases; correct: 3384.

I0117 11:15:15.194531 139934521526016 train.py:42] accuracy:  91.28%; precision:  89.10%; recall:  88.63%; FB1:  88.87

I0117 11:15:15.194565 139934521526016 train.py:42]                  : precision:  89.10%; recall:  88.63%; FB1:  88.87  3798

I0117 11:15:15.195159 139934521526016 train.py:38] evaluate:test
I0117 11:15:15.936123 139934521526016 train.py:42] processed 6661 tokens with 3682 phrases; found: 3649 phrases; correct: 3442.

I0117 11:15:15.936207 139934521526016 train.py:42] accuracy:  95.48%; precision:  94.33%; recall:  93.48%; FB1:  93.90

I0117 11:15:15.936244 139934521526016 train.py:42]                  : precision:  94.33%; recall:  93.48%; FB1:  93.90  3649

```

### 模型推理
#### 参数设置
```bash
#!/usr/bin/env bash
python inference.py \
    --init_checkpoint chinese_L-12_H-768_A-12 \  # 获取vocab.txt
    --max_seq_len 128 \  # 句子最长长度
    --output output  # 模型路径
```
#### 开始推理
```bash
./inference.sh
```
#### 推理日志
> 以分词为例
```text
中国你好成都
{'string': ['中', '国', '你', '好', '成', '都'], 'entities': [{'word': '中国', 'start': 0, 'end': 2, 'type': 'cut'}, {'word': '成都', 'start': 4, 'end': 6, 'type': 'cut'}]}
```
#### saved_model格式模型导出
> 推理结束后，会自动导出saved_model格式模型，用于部署。
### 模型部署
#### 模型文件
```text
saved_model
└── 000000
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

2 directories, 3 files
```
#### docker部署
```bash
#!/usr/bin/env bash
docker run -t --rm -p 8501:8501 \
-v "$(pwd)/saved_model:/models/docker_test" \
-e MODEL_NAME=docker_test tensorflow/serving
```
#### 客户端依赖文件
```text
vocab
├── maps.pkl   # 标签对应表
├── trans.npy  # 转移概率矩阵
└── vocab.txt  # 词对应表

0 directories, 3 files

```
#### 客户端测试
```python
from client import get_result
text = "中国你好成都。"
res = get_result(text)
print(res)
```

#### 测试效果
```text
{'string': ['中', '国', '你', '好', '成', '都'], 'entities': [{'word': '中国', 'start': 0, 'end': 2, 'type': 'cut'}, {'word': '成都', 'start': 4, 'end': 6, 'type': 'cut'}]}
```
