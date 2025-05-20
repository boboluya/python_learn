# 数据集
import os
import pandas


def get_box(str: str) -> list:
    strs = str.split(" ")
    strs = [int(i) for i in strs]
    return strs[:4]


def get_datas():

    #  训练集
    y_path_train = r"C:\Users\bobol\Desktop\CNN\faces\train\wider_face_train_bbx_gt.txt"
    x_path_train = r"C:\Users\bobol\Desktop\CNN\faces\train"
    with open(y_path_train, "r") as f:
        train_lines = f.readlines()
    train_i = 0
    y_train = []
    x_train = []
    maxc = 0
    while train_i < len(train_lines):
    # while train_i < 200:
        file_name = train_lines[train_i].strip()
        faces = train_lines[train_i + 1]
        face_count = int(faces)
        if face_count == 0:
            train_i += 3
            continue
        else:
            a = []
            index = train_i + 2
            while index <= train_i + 1 + face_count:
                box_str = train_lines[index].strip()
                a.append(get_box(box_str))
                index += 1
            # 不能无脑取a0，这里改成取面积最大的
            si = 0
            si2 = 0
            for aa in range(len(a)):
                si21 = a[aa][2] * a[aa][3]
                if si21 > si2:
                    si = aa
                    si2 = si21
            y_train.append(a[si])
        x_train.append(os.path.join(x_path_train, file_name))
        train_i += 2
        if face_count == 0:
            face_count += 1
        train_i += face_count

    # 验证集
    y_path_val = r"C:\Users\bobol\Desktop\CNN\faces\val\wider_face_val_bbx_gt.txt"
    x_path_val = r"C:\Users\bobol\Desktop\CNN\faces\val"
    with open(y_path_val, "r") as fv:
        val_lines = fv.readlines()
    val_i = 0
    y_val = []
    x_val = []
    while val_i < 100:
        file_name = val_lines[val_i].strip()
        file_name.replace("/", "\\")
        faces = val_lines[val_i + 1]
        face_count = int(faces)
        if face_count == 0:
            val_i += 3
            continue
        else:
            a = []
            index = val_i + 2
            while index <= val_i + 1 + face_count:
                box_str = val_lines[index].strip()
                a.append(get_box(box_str))
                index += 1
            y_val.append(a[0])
        x_val.append(os.path.join(x_path_val, file_name))
        val_i += 2
        if face_count == 0:
            face_count += 1

        val_i += face_count
    fr = {"a": x_train, "b": y_train}
    print(y_train[0])

    #

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    print("ssss")
    get_datas()
