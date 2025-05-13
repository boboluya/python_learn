import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("C:\\Users\\bobol\\Desktop\\CNN\\test.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.blur(image,(3,3),(3,3))



# image = cv2.resize(image,(20,20))
# image,b,r=cv2.split(image)
# print(image)
# imager=cv2.getRotationMatrix2D((10,10),200,1)
# print(imager)
# image=cv2.warpAffine(image,imager,(20,20))
# 平移
# translation_matrix = np.float32([[1, 0, 1], [0, 1, 1]])  # tx, ty 为平移距离
# translated_image = cv2.warpAffine(image, translation_matrix, (20, 20))
# # print(translated_image)
# flipped_image = cv2.flip(image, 0)

# image = cv2.add(image/10,b/10)
# print(image)
# image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# image=cv2.GaussianBlur(image,(5,5),0)
# edges = cv2.Canny(image, 200, 200)
# b,r,g=cv2.split(image)
# r=r*0
# image=cv2.merge([b,r,g])
# 计算 x 方向的梯度
# 应用 Laplacian 算子
# laplacian = cv2.Laplacian(image, cv2.CV_64F)
#
# # 显示结果
# cv2.imshow('Laplacian', laplacian)

# 二值化处理
# _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# # 查找轮廓
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# 创建一个空白图像
# output = np.zeros_like(image)
# for contour in contours:
#     epsilon = 0.01 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
# # cv2.imshow('Approx Polygons', binary)
# cv2.waitKey(0)
# 计算直方图
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#
# # 绘制直方图
# plt.plot(hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()
# 创建 VideoCapture 对象，读取摄像头视频
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# 读取视频帧
while True:
    ret, frame = cap.read()

    # 如果读取到最后一帧，退出循环
    if not ret:
        break
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    # _, binary = cv2.threshold(frame, 127, 200, cv2.THRESH_BINARY)
    # cv2.imshow('Min Area Rectangles', binary)
    # 显示当前帧
    # cv2.imshow('Camera', output)

    # faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #
    # # 在帧上绘制矩形框标记人脸
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 计算当前帧与前一帧的差异
    # frame_diff = frame - prev_gray

    # # 对差异图像进行二值化处理
    # _, thresh = cv2.threshold(frame, 200, 244, cv2.THRESH_BINARY)
    # k,v = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # f2=np.zeros_like(thresh)
    # cv2.drawContours(f2,k,-1,(255,0,0),2)
    # cv2.imshow("frame", f2)
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])

    # frame=cv2.filter2D(frame, -1, kernel)

    frame =cv2.Canny(frame,100,200)
    cv2.imshow('frame', frame)

    # frame=cv2.Canny(frame,100,150)
   # 更新前一帧

    # 按下 'q' 键退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()










# cv2.imshow("image",image)

# key=cv2.waitKey(0)
#
# save_path="C:\\Users\\bobol\\Desktop\\CNN\\test2.jpg"
# cv2.imwrite(save_path,image)
# cv2.destroyAllWindows()