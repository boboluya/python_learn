import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# ----------------------
# 1. 数据准备
# ----------------------

# 加载标签和构建路径列表
img_dir = "./your_images_folder"
json_path = "./labels.json"

# 加载图片和分类对应的JSON文件
with open(json_path) as f:
    labels_dict = json.load(f)

# 过滤有效图片并构建路径标签列表
img_paths = []
labels = []
for fname in os.listdir(img_dir):
    if fname.endswith((".jpg", ".png")) and fname in labels_dict:
        img_paths.append(os.path.join(img_dir, fname)) # 图片地址列表
        label_str = labels_dict[fname].lower()
        labels.append(0 if label_str == "cat" else 1) # 图片类别列表

# 数据集参数
dataset_size = len(img_paths)   # 数据集容量
train_size = int(0.8 * dataset_size) # 训练容量
val_size = dataset_size - train_size # 测试容量
batch_size = 32 #每批数量

# 创建TensorFlow数据集
def load_and_preprocess(img_path, label):
    # 读取图片
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) #解码，按照三通道
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0-1范围？？？？？？？？？？？？？？？是归一化操作，因为首先解码后图像是int类型的，而tf或者keras要求输入是float32的；其次归一化减少数值防止梯度爆炸；复合激活函数的预期输入；后续标准化必须要求先归一化。
    
    # 数据增强
    img = tf.image.resize(img, [128, 128])  # 重新设置分辨力
    img = tf.image.random_flip_left_right(img)  # 随机裁剪
    img = tf.image.random_rotation(img, 15 * np.pi / 180)  # 弧度制；随机旋转
    
    # 标准化 (ImageNet)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]     #标准化消除了通道偏差，避免了某个通道（比如红色通道）过于主导模型；提高训练效率，因为标准化后的梯度更加缓和，有利于模型收敛。
    return img, label

# 创建完整数据集并打乱
dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels)) #？？？？？？原来是tf自带的把两个数组一一对应合并成map对象，理由后续操作
dataset = dataset.shuffle(buffer_size=dataset_size) # 打乱顺序？？？？避免随机性差。1、在数据集中加载到内存缓冲区，buffer_size就是缓冲区大小，拿走一个补充一个；
dataset = dataset.map(load_and_preprocess)  # 自动将map拆包传入到相应方法执行数据处理，方法里必须是纯粹tf操作，加入其它参数能做到并行优化

# 划分训练集和验证集
train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)# 异步预加载​​数据，实现“数据准备”和“模型训练”的重叠，最大化GPU利用率。
val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# 高效数据管道：take\skip->batch->prefetch(tf.data.autotune)// 划分训练集->批量训练->消除io瓶颈


# ----------------------
# 2. 定义模型
# ----------------------

class CatDogClassifier(Model):
    def __init__(self):
        super().__init__()
        
        # 特征提取
        self.features = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2)
        ])
        
        # 分类器
        self.classifier = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1)  # 输出logits
        ])
    
    def call(self, x):
        x = self.features(x)
        return self.classifier(x)

# 初始化模型
model = CatDogClassifier()

# ----------------------
# 3. 训练配置
# ----------------------

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # from_logits=True代表自动使用Sigmoid函数，意思是自动处理为0,1概率问题。另外直接使用logits避免极大极小值时候仍能保持稳定
# 交叉熵在误差大的时候梯度大，加快训练。均方误差的梯度在接近01时会出现梯度消失
#均方误差MSE适合在回归任务中使用，预测房价气温等
# 生成式模型仍然是使用的交叉熵，因为它本质上仍是分类问题，只不过是从千万个子里面找到概率最高的字。


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
'''
此处使用了adam优化器，学习率是0.001，此学习率在分类任务重正合适，过大导致震荡，过小导致缓慢
为什么使用adam优化器?
1、分类任务具有稀疏性和离散性，adam优化器能对于离散参数设置独立的学习率，使得稀疏参数可以得到更好的学习加快训练，密集参数减少训练避免学习导致的震荡。
2、adam优化器具有动量机智，可以减少局部极值带来的局部最优解。
参数是
'''

best_val_acc = 0.0

# ----------------------
# 4. 训练循环
# ----------------------

num_epochs = 10

for epoch in range(num_epochs):
    # 训练阶段
    train_loss = 0.0
    for images, labels in train_dataset:    # train_dataset是训练集，一个批次的训练集
        with tf.GradientTape() as tape: #   GradientTape是传播磁带，意思是记录传播过程中的梯度，以便于后续的反向传播计算
            logits = model(images, training=True)   #把图片和分类数据map输入到模型里，返回的logits是一个（二分类：标量）（多分类：向量），未经过Sigmoid和softmax的，training是否为true会影响一些行为。未知。
            loss = loss_fn(labels, logits)  # 损失函数计算损失值，输入为（理想结果p,预测结果y),比如此处是二分类交叉熵损失函数p-y，因为loss_fn是二分类交叉熵损失，并且启用了from_logits=TRUE，这会在方法里自动调用Sigmoid函数，请不要手动调用。多分类任务为softmax。
        grads = tape.gradient(loss, model.trainable_variables)  # tape.gradient是计算梯度的方法，具体到数学上来说就是∂H/∂z，计算梯度值，参数是（损失值，模型的参数-就是可以被训练的参数值），返回是梯度
        optimizer.apply_gradients(zip(grads, model.trainable_variables))    # zip是把两个list一一对应放到map里再放到一个list；apply_gradients是执行参数优化，入参是梯度和参数列表。内部实现比较复杂。
        train_loss += loss * images.shape[0]    # 计算累积的损失，本批次的
    
    # 验证阶段
    val_loss = 0.0
    correct = 0
    for images, labels in val_dataset:
        logits = model(images, training=False)
        val_loss += loss_fn(labels, logits) * images.shape[0]
        
        # 计算准确率
        probs = tf.sigmoid(logits)
        preds = tf.cast(probs > 0.5, tf.int32)
        correct += tf.reduce_sum(tf.cast(preds == labels[:, tf.newaxis], tf.int32))
    
    # 统计指标
    train_loss = train_loss / train_size
    val_loss = val_loss / val_size
    val_acc = correct.numpy() / val_size
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights("best_model_tf.h5")
    
    # 学习率衰减
    if (epoch + 1) % 5 == 0:
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)
    
    # 打印日志
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"训练完成！最佳验证准确率: {best_val_acc * 100:.2f}%")