import tensorflow as tf
import json
import os
import numpy as np
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 配置参数
IMG_SIZE = 150
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
JSON_PATH = "labels.json"  # 你的JSON文件路径
IMAGE_DIR = "images"  # 图片存储目录

# 加载并解析JSON标注
with open(JSON_PATH) as f:
    annotations = json.load(f)

# 示例JSON结构：
# [
#   {"file": "img_001.jpg", "label": "cat"},
#   {"file": "img_002.jpg", "label": "dog"},
#   ...
# ]

# 构建路径列表和标签列表
file_paths = [os.path.join(IMAGE_DIR, item["file"]) for item in annotations]
labels = [1 if item["label"] == "dog" else 0 for item in annotations]  # 猫:0 狗:1

# 划分数据集（80%训练，10%验证，10%测试）
X_train, X_temp, y_train, y_temp = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# 创建TensorFlow Dataset对象-------数据管道
def create_dataset(file_paths2, labels2):
    return tf.data.Dataset.from_tensor_slices((file_paths2, labels2))   #这会把两个数组加载成map，当然不是py自带的字典


train_ds = create_dataset(X_train, y_train)
val_ds = create_dataset(X_val, y_val)
test_ds = create_dataset(X_test, y_test)


# 数据预处理函数
def process_path(file_path, label):
    # 加载图像
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # 调整大小和归一化
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# 数据增强层
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.2),
    ]
)


# 构建数据管道
def prepare_dataset(ds, shuffle=False, augment=False):
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(buffer_size=AUTOTUNE)


# 应用预处理
train_ds = prepare_dataset(train_ds, shuffle=True, augment=True)
val_ds = prepare_dataset(val_ds)
test_ds = prepare_dataset(test_ds)


# 构建模型（使用预训练模型提升效果）
def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # 冻结基础模型

    return tf.keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


model = create_model()

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 训练参数
EPOCHS = 15

# 定义回调,
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, verbose=1),    # 设置早停，早停时机是在三个epoch没有提升的时候，patience=n个epoch，verbose=1输出提示
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),   # 模型保存点，在验证集上比之前的表现更好就保存一个h5模型
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),    # reduce减少，LR学习率，OnPlateau在没有明显性能提升时，factor=学习率减小程度，patience=2次epoch没有提升，verbose=显示提示
]

# 训练模型
history = model.fit(
    train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks
)

# 评估模型
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.2%}")

# 保存完整模型
model.save("cat_dog_classifier_v2.h5")


# 预测函数示例
def predict_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.expand_dims(img, 0) / 255.0

    pred = model.predict(img)
    return "Dog" if pred[0] > 0.5 else "Cat"


# 使用示例
# print(predict_image("test_image.jpg"))
