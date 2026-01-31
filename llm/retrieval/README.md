## 基于大模型的图像检索
一个最基本的基于大模型的图像离线检索方案demo。核心点包括两部分：特征提取和相似度计算。

### Install
```
pip install sentence_transformers=5.0.0
```

### 特征提取
特征提取模型：[GME](https://arxiv.org/pdf/2412.16855)

使用`sentence_transformers`推理，embedding保存至文件名命令的`.npy`文件中

使用示例：
```bash
python inference -c iic/gme-Qwen2-VL-2B-Instruct \
    -d data/images \
    -o data/embeddings
```

### 相似度计算
相似度计算方法：余弦相似度

加速计算：GPU+并行

使用示例：
```bash
python retrieval.py --l data/embeddings \
    -q data/embeddings_query \
    -o data/final.txt
```
