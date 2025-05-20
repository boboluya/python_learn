# 训练模型
import tensorflow as tf
import date_sets as ds

if __name__=="__main__":
    t, v = ds.get_datasets()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 1)),
        tf.keras.layers.MaxPooling2D(2),
        # tf.keras.layers.Conv2D(64, 3, activation='relu'),
        # tf.keras.layers.MaxPooling2D(2),
        # tf.keras.layers.Conv2D(128, 3, activation='relu'),
        # tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # 指标
    # metrics = tf.keras.metrics.MeanAbsoluteError(name='mae')
    
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics = ["mae"]
    )
    
    model.fit(t,epochs=2)
    

    model.save("./face_get22.keras")

    model.summary()

