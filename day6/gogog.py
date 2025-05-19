import tensorflow as tf
import cv2
import numpy as np
import train

# 加载模型
model = tf.keras.models.load_model("face_get22.keras", custom_objects={"FaceExistClass": train.FaceExistClass})

# 读取图片并预处理
img_path = r"C:\Users\bobol\Desktop\CNN\WIN_20250519_20_00_15_Pro.jpg"
img_color = cv2.imread(img_path)  # 彩色图，稍后用来画框
# img_color = cv2.resize(img_color, (128, 128))
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (128, 128))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

# 模型预测
y_pred = model.predict(img)[0]  # 形状是 [1, 4]，取出一维

# 转为整数坐标
x, y, w, h = [int(i) for i in y_pred]

# 在彩色图上画框
cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示图像
cv2.imshow("Prediction", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
