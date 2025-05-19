# 训练模型
import tensorflow as tf
import date_sets as ds




class FaceExistClass(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)

        self.cv2d1 = tf.keras.layers.Conv2D(
            16, 5, activation="relu", input_shape=(128, 128, 1)
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.flat = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(4)

    def call(self, x):
        x = self.cv2d1(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




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


if __name__=="__main__":
    t, v = ds.get_datasets()
    
    model = FaceExistClass()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # 指标
    metrics = tf.keras.metrics.MeanAbsoluteError(name='mae')
    

    epoch = 0

    while epoch < 2:
        for x, y in t:
            train_m(x, y)
        print(f"Epoch {epoch+1}, Acc: {metrics.result():.4f}")
        metrics.reset_state()
        epoch += 1


    # print(t)
    # model.fit(t, validation_data=v, epochs=2)  # 训练集 Dataset  # 验证集 Dataset

    model.save("./face_get22.keras")

    model.summary()

