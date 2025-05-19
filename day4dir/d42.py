import tensorflow as tf

# 分类损失，标准二分类交叉熵
def classification_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# 带掩码的回归损失（只计算正样本的边框）
def masked_smooth_l1_loss(y_true, y_pred):
    # y_true 第1维是分类标签，shape = (batch_size, 5)
    # 前1个是分类标签，后4个是边框坐标
    cls_label = y_true[:, 0]  # 0或1
    bbox_true = y_true[:, 1:] # 4维边框

    diff = tf.abs(bbox_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = less_than_one * 0.5 * diff**2 + (1 - less_than_one) * (diff - 0.5)
    loss = tf.reduce_sum(loss, axis=1)  # 每个样本回归损失和

    # 只对正样本（cls_label=1）计算回归损失
    mask = tf.cast(tf.equal(cls_label, 1), tf.float32)
    loss = loss * mask

    # 平均回归损失，除以正样本数，防止梯度爆炸或消失
    num_positive = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(loss) / num_positive

# 组合总损失函数
def multitask_loss(y_true, y_pred):
    # y_true, y_pred均为列表 [cls, reg]
    cls_true, reg_true = y_true
    cls_pred, reg_pred = y_pred

    # 分类损失
    cls_loss = classification_loss(cls_true, cls_pred)

    # 拼接回归输入为 (batch_size, 5)，第一列是cls标签，后4列是bbox
    reg_true_combined = tf.concat([cls_true, reg_true], axis=1)
    reg_loss = masked_smooth_l1_loss(reg_true_combined, reg_pred)

    return cls_loss + reg_loss
