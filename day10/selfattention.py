import tensorflow as tf
import os

# 超参数
VOCAB_SIZE = 10000  # 词汇表大小
MAX_LEN = 100       # 最大序列长度
EMBED_DIM = 64      # 词向量维度
NUM_HEADS = 8       # 多头注意力头数
BATCH_SIZE = 32     # 批次大小
EPOCHS = 5          # 训练轮数

# 多头注意力层定义
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # 保证可以均分
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度

        # 定义用于生成Q、K、V的全连接层
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # 输出层
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 将最后一个维度分割成(num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # 调整维度顺序为(batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        # 通过全连接层生成Q、K、V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 拆分多头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 计算Q和K的点积注意力分数
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # 缩放

        if mask is not None:
            scaled_attention_logits += mask * -1e9  # 掩码处理

        # 归一化得到注意力权重
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # 加权求和得到输出
        output = tf.matmul(attention_weights, v)

        # 还原维度顺序
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        # 合并多头
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        # 通过输出层
        output = self.dense(concat_attention)
        return output, attention_weights

# 简单文本数据加载函数，假设格式：每行一句话，下一行标签
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        for i in range(0, len(lines), 2):
            texts.append(lines[i])              # 读取文本
            labels.append(int(lines[i + 1]))    # 读取标签
    return texts, labels

# 加载训练和验证数据
train_texts, train_labels = load_data("/root/train.txt")
val_texts, val_labels = load_data("/root/val.txt")

# 文本向量化层，将文本转为整数序列
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN,
)
vectorize_layer.adapt(train_texts)  # 适配训练集

# 构造数据集
def prepare_dataset(texts, labels):
    texts = vectorize_layer(tf.constant(texts))  # 文本转为整数序列
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = prepare_dataset(train_texts, train_labels)
val_ds = prepare_dataset(val_texts, val_labels)

# 构建文本分类模型
class TextClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)  # 词嵌入层
        self.mha = MultiHeadAttention(embed_dim, num_heads)                # 多头注意力层
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()    # 全局平均池化
        self.dropout = tf.keras.layers.Dropout(0.2)                        # Dropout防止过拟合
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")   # 输出层（二分类）

    def call(self, x):
        x = self.embedding(x)              # (batch, seq_len, embed_dim)
        attn_output, _ = self.mha(x, x, x) # 自注意力
        x = self.global_avg_pool(attn_output)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# 实例化和编译模型
model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 训练模型
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
