import tensorflow as tf

# Smooth L1 损失（Huber Loss）
def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = less_than_one * 0.5 * diff**2 + (1 - less_than_one) * (diff - 0.5)
    return tf.reduce_mean(loss)

# 多任务模型定义
def build_face_detection_model(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Backbone: 简单卷积层提取特征
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # 输出变成一维特征向量

    # 分类头 - 输出是否有人脸的概率
    cls_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_output')(x)

    # 回归头 - 输出边界框坐标 [x, y, w, h]
    reg_output = tf.keras.layers.Dense(4, activation='linear', name='reg_output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[cls_output, reg_output])
    return model

# 构建模型
model = build_face_detection_model()

# 编译模型，指定多任务损失和权重
model.compile(
    optimizer='adam',
    loss={
        'cls_output': 'binary_crossentropy',
        'reg_output': smooth_l1_loss
    },
    loss_weights={'cls_output': 1.0, 'reg_output': 1.0},
    metrics={'cls_output': 'accuracy'}
)

model.summary()
