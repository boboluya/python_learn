import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model("number_get.h5")


# 获取预测概率分布（每张图属于各类别的概率）
y_pred_probs = model.predict(x_test)

# 取每一行最大概率的索引 → 得到预测的标签
y_pred = np.argmax(y_pred_probs, axis=1)

# 找出预测值 ≠ 真实值的位置
wrong_indices = np.where(y_pred != y_test)[0]

print(f"模型一共错了 {len(wrong_indices)} 张图片")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 用 sklearn 自带的方法画出混淆矩阵图
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[str(i) for i in range(10)]
)

plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Reds)
plt.title("MNIST 测试集混淆矩阵")
plt.show()
