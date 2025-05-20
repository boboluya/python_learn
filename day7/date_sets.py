# 数据集转换成数据管道
import tensorflow as tf
import datas_fun as df
import resize as rz
import matplotlib.pyplot as plt
import cv2


def show_images(images, labels, n=4):
    for i in range(n):
        img = images[i].numpy().squeeze()  # (128,128,1) -> (128,128)
        label = labels[i].numpy()  # 假设标签是框坐标+类别，格式需确认
        x, y, w, h = [int(i * 256) for i in label]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("s",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_datasets():
    xt, yt, xv, yv = df.get_datas()
    # 转换成 RaggedTensor（支持不同图像中有不同数量的框）
    # yt = tf.ragged.constant(yt, dtype=tf.int32)
    # yv = tf.ragged.constant(yv, dtype=tf.int32)

    # 缩放标签
    for i in range(len(xt)):
        # print([xt[i],yt[i]])
        yt[i] = rz.resizeimg(xt[i], yt[i], [256, 256])
    for i in range(len(xv)):
        yv[i] = rz.resizeimg(xv[i], yv[i], [256, 256])

    t = get_datasets2(xt, yt)
    # for images, labels in t.take(1):
    #     show_images(images, labels, n=4)
    v = get_datasets2(xv, yv)
    count = 0
    for batch in t:
        images, labels = batch
        count += images.shape[0]  # 每个 batch 的实际图片数
    print("总图片数:", count)
    for n, m in t:
        print
    return t, v


def set_y(x, y):
    for i in range(len(x)):
        print(x.shape())


def get_img_map(x, y):
    img = tf.io.read_file(x)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    orig_shape = tf.shape(img)[:2]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 数据增强
    img = tf.image.resize(img, [256, 256])

    return img, y


def get_datasets2(x, y):
    # print(y)
    w = tf.data.Dataset.from_tensor_slices((x, y))
    v = w.map(get_img_map).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return v


if __name__ == "__main__":
    get_datasets()
