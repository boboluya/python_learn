import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# _, pre_frame = cap.read()
# pre_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
# 读取视频帧
while True:
    ret, frame = cap.read()
    # 如果读取到最后一帧，退出循环
    if not ret:
        break
    #
    # # 将帧转换为灰度图像
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # 计算当前帧与前一帧的差异并且在差异处绘制矩形框
    # diff = cv2.absdiff(gray, pre_gray)
    # _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     if cv2.contourArea(contour) > 0:  # 过滤掉小的噪声
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # # 更新前一帧
    # pre_gray = gray.copy()

    b,r,g= cv2.split(frame)
    zeros=np.zeros_like(b)
    # 交换颜色通道
    frame = cv2.merge([zeros,b,b-100])

    # 显示当前帧
    cv2.imshow('Camera', frame)
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()
