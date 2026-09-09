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

## 1. 预处理/平台算法

### 食品相关性聚类
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

---

## 2. 多模态情感分析模型

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



### OPT-style 图-文-音预训练
- Text / Vision / Audio Encoder
- Cross-Modal Transformer
- Masked Language Modeling
- Masked Vision Modeling
- Masked Audio Modeling

`OPTStyleTriModal.masked_pretrain_loss()` 可直接跑三模态 masking 预训练。

另外 `TinyMLMNSP` 单独实现了 MLM + NSP。

---

## 3. 持续学习：经验重放

`GradientReplayBuffer` 代码实现：

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

`grouped_gradient_signature()` ：

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

流程是：

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

项目采用：
- 文本随机 token 替换
- 图像/音频特征加高斯噪声

这部分是明确的工程补全项。

---

## 5. 随机合成数据

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

