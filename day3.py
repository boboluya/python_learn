# 训练识别手写数字

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train = tf.image.random_flip_left_right(x_train)
# x_train = tf.image.random_contrast(x_train)
# x_train = tf.image.random_flip_up_down(x_train)

class_weight = {i: 1.0 for i in range(10)}  # 默认权重都为1



model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=32, class_weight=class_weight)
model.evaluate(x_test, y_test, verbose=2)
model.save("number_get.h5")
model.summary()

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率: {test_acc:.4f}，损失: {test_loss:.4f}")
