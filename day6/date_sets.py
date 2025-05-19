# 数据集转换成数据管道
import tensorflow as tf
import datas_fun as df


def get_datasets():
    xt, yt, xv, yv = df.get_datas()
    # 转换成 RaggedTensor（支持不同图像中有不同数量的框）
    # yt = tf.ragged.constant(yt, dtype=tf.int32)
    # yv = tf.ragged.constant(yv, dtype=tf.int32)
    t=get_datasets2(xt,yt)
    v=get_datasets2(xv,yv)
    count = 0
    for batch in t:
        images, labels = batch
        count += images.shape[0]  # 每个 batch 的实际图片数
    print("总图片数:", count)
    for n,m in t:
        print
    return t,v

def set_y(x,y):
    for i in range(len(x)):
        print(x.shape())

def get_img_map(x, y):
    img = tf.io.read_file(x)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.rgb_to_grayscale(img) 
    orig_shape = tf.shape(img)[:2]
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # 数据增强
    img = tf.image.resize(img,[128,128])
    
    return img,y


def get_datasets2(x, y):
    # print(y)
    w=tf.data.Dataset.from_tensor_slices((x, y))
    v=w.map(get_img_map).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return v

if __name__ == "__main__":
    get_datasets()
