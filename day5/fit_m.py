# 训练模型
import tensorflow as tf
import data_sets as ds

t, v = ds.get_datasets()


class FaceExistClass(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.cv2d1 = tf.keras.layers.Conv2D(
            16, 5, activation="relu", input_shape=(128, 128, 1)
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.flat = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.cv2d1(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        return x


model = FaceExistClass()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()
# 指标
metrics = tf.keras.metrics.BinaryAccuracy()


@tf.function
def train_m(x, y):
    with tf.GradientTape() as tg:
        logits = model(x)
        loss2 = loss_fn(y, logits)
    grads = tg.gradient(loss2, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # 更新准确率
    metrics.update_state(y, logits)

    return loss2


epoch = 0

while epoch < 2:
    for x, y in t:
        train_m(x, y)
    print(f"Epoch {epoch+1}, Acc: {metrics.result():.4f}")
    metrics.reset_state()
    epoch += 1


# print(t)
# model.fit(t, validation_data=v, epochs=2)  # 训练集 Dataset  # 验证集 Dataset
test_loss, test_acc = model.evaluate(v)
print(f"测试集准确率: {test_acc:.4f}，损失: {test_loss:.4f}")
model.save("face_get.h5")

model.summary()
