import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("face_get.h5")

cap = cv2.VideoCapture(0)

k,v = cap.read()
img=cv2.cvtColor(v,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (128, 128))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=-1)  # (200, 200, 1)
img = np.expand_dims(img, axis=0)   # (1, 200, 200, 1) — batch size 为 1

y_pred_probs = model.predict(img)
print(y_pred_probs)