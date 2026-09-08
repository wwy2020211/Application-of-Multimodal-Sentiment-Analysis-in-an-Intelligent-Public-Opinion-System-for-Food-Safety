# 毕设答辩算法复现工程

本工程根据上传的《毕业答辩》PPT，把答辩中出现的主要算法拆成独立模块，
并提供随机/合成多模态数据，使项目无需企业私有数据也能跑通。

> 重要：PPT并没有给出所有网络的完整层数、预训练 checkpoint、超参数和
> 企业内部公式。本工程对“PPT明确给出的结构/伪代码”尽量逐项实现；对只出现
> 名称而没有细节的算法采用**标准定义或明确标注的 proxy**，不会把推测写成
> “原作者精确实现”。

## 工程结构

```text
thesis_defense_repro/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/defense_repro/
│   ├── data.py
│   ├── metrics.py
│   ├── train.py
│   ├── preprocessing/
│   │   ├── video.py                # 灰度变换 + 图像平滑
│   │   ├── audio.py                # FFT
│   │   ├── text.py                 # Skip-Gram Word2Vec
│   │   └── relevance.py            # 食品相关/不相关聚类（K-means重构）
│   ├── platform/
│   │   ├── single_pass.py          # Single-pass 新闻聚类
│   │   ├── tfidf.py                # TF-IDF
│   │   └── heat_index_proxy.py     # 网络传播热度 proxy（PPT无公式）
│   ├── models/
│   │   ├── ef_lstm.py              # EF-LSTM
│   │   ├── lf_lstm.py              # LF-LSTM
│   │   ├── tfn.py                  # Tensor Fusion Network
│   │   ├── unimodal.py             # only vision/audio/language
│   │   ├── text_baselines.py       # LSTM / CNN / SVM
│   │   ├── prompt_model.py         # 图-文 Prompt 模型及消融
│   │   ├── opt_style.py            # 图-文-音 OPT-style Transformer
│   │   └── pretrain.py             # MLM + NSP
│   ├── continual/
│   │   ├── gradient_replay.py      # 第19页经验重放伪代码
│   │   └── gradient_fusion.py      # G_multi=aG_text+bG_img+cG_downstream
│   └── active/
│       └── robustness.py           # 基于鲁棒性的主动学习 QUERY
├── scripts/
│   ├── demo_preprocessing.py
│   ├── demo_sentiment.py
│   ├── demo_opt_pretrain.py
│   ├── demo_continual.py
│   ├── demo_active.py
│   └── demo_all.py
└── tests/
    └── test_smoke.py
```

## 1. PPT中预处理/平台算法

### 食品相关性聚类
PPT只写“用聚类法对数据进行食品相关与不相关二分类”，没有指定聚类器。
工程使用二维 K-means 作为可运行重构。

### 视频
- RGB灰度变换
- 均值平滑

### 音频
- `torch.fft.rfft`

### 文本
- one-hot/token ID
- Skip-Gram Word2Vec

### Single-pass
实现在线新闻聚类：新样本与现有簇中心做 cosine similarity，超过阈值则加入，
否则创建新簇。

### TF-IDF
纯 Python 实现。

### 网络传播热度
PPT只出现“网络传播热度指数算法”名字，没有任何公式。因此
`heat_index_proxy.py` 是显式标注的可配置 proxy，不应当称为论文原公式。

---

## 2. 多模态情感分析模型

PPT结果表中出现的模型都给了可运行版本：

- `EFLSTM`
- `LFLSTM`
- `TensorFusionNetwork`
- `UnimodalClassifier("vision")`
- `UnimodalClassifier("audio")`
- `UnimodalClassifier("language")`
- `TextLSTM`
- `TextCNN`
- `LinearSVM`
- `PromptImageTextClassifier`
- `OPTStyleTriModal`

TFN采用真正的三阶 tensor fusion：

```python
z = torch.einsum("bi,bj,bk->bijk", t, v, a)
```

### 图-文 Prompt

PPT结构：
- text + `This message is [EMO]`
- TXT encoder
- `[EMO] -> softmax loss`
- TXT feature + IMG feature -> auxiliary softmax loss

项目支持三种消融：

```text
clip_prompt       use_prompt=True,  finetune=True
without_prompt    use_prompt=False, finetune=True
without_finetune  use_prompt=True,  finetune=False
```

注意：PPT没有给出实际 CLIP checkpoint，因此这是结构复现，不是假装下载了原模型。

### OPT-style 图-文-音预训练

根据PPT图：
- Text / Vision / Audio Encoder
- Cross-Modal Transformer
- Masked Language Modeling
- Masked Vision Modeling
- Masked Audio Modeling

`OPTStyleTriModal.masked_pretrain_loss()` 可直接跑三模态 masking 预训练。

另外 `TinyMLMNSP` 单独实现了PPT背景页提到的 MLM + NSP。

---

## 3. 持续学习：经验重放

`GradientReplayBuffer` 按PPT第19页伪代码实现：

```text
c = max_i cosine(g, G_i) + 1
if memory full:
    if c < 1:
        i ~ C_i / sum C_j
        r ~ Uniform(0,1)
        if r < C_i/(C_i+c):
            M_i <- (x,y)
            C_i <- c
else:
    append
```

`grouped_gradient_signature()` 对应第20页：

```text
G_multimodal =
    a G_textencoder
  + b G_imageencoder
  + c G_downstreamnet
```

由于不同参数组位于不同坐标块，代码用“加权后拼接梯度块”构造统一 gradient
signature，再计算 cosine similarity。

---

## 4. 主动学习：鲁棒性 QUERY

PPT流程是：

```text
无标注数据
   + 扰动后的无标注数据
       ↓
      模型
       ↓
  结果是否变化？
       ↓ Yes
  加入人工标注池
```

`RobustnessQueryStrategy`严格保留“预测类别是否变化”为最高优先级，
并用 Jensen-Shannon divergence 对发生变化的样本排序。

PPT没有给出具体扰动方法，因此项目采用：
- 文本随机 token 替换
- 图像/音频特征加高斯噪声

这部分是明确的工程补全项。

---

## 5. 随机合成数据

企业食品安全数据并未随答辩文件公开，所以项目默认使用可学习的随机合成数据：

```text
text   [B,T] token IDs
vision [B,T,Dv]
audio  [B,T,Da]
label  positive / negative / neutral
```

三种模态共享类别 latent signal，因此模型可以训练而不是纯随机猜测。

---

## 安装

```bash
cd thesis_defense_repro
pip install -e .
```

## 运行预处理

```bash
python scripts/demo_preprocessing.py
```

## 单独跑情感模型

```bash
python scripts/demo_sentiment.py --model ef_lstm
python scripts/demo_sentiment.py --model lf_lstm
python scripts/demo_sentiment.py --model tfn
python scripts/demo_sentiment.py --model only_vision
python scripts/demo_sentiment.py --model only_audio
python scripts/demo_sentiment.py --model only_language
python scripts/demo_sentiment.py --model clip_prompt
python scripts/demo_sentiment.py --model without_prompt
python scripts/demo_sentiment.py --model without_finetune
python scripts/demo_sentiment.py --model opt_finetune
python scripts/demo_sentiment.py --model text_lstm
python scripts/demo_sentiment.py --model text_cnn
python scripts/demo_sentiment.py --model svm
```

## 三模态 masking 预训练

```bash
python scripts/demo_opt_pretrain.py
```

## 持续学习经验重放

```bash
python scripts/demo_continual.py
```

## 主动学习

```bash
python scripts/demo_active.py
```

## 全部 smoke demo

```bash
python scripts/demo_all.py
```

## 论文/答辩未给出的关键信息

以下内容PPT不足以精确恢复，所以项目没有伪造“原始精确值”：

1. 企业训练集原始文件与划分；
2. CLIP/OPT具体 checkpoint、tokenizer、预训练数据；
3. image/text/audio encoder 的精确层数和隐藏维度；
4. prompt verbalizer / label word 的完整定义；
5. 主动学习的具体扰动算子；
6. “食品相关二分类”具体聚类算法；
7. 网络传播热度指数的公式；
8. 大部分训练超参数。

这些位置都在代码注释里标明了“standard reconstruction”或“proxy”。
