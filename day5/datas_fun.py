# 数据集
import os
import pandas


def get_datas():

    #  训练集
    y_path_train = r"C:\Users\bobol\Desktop\CNN\faces\train\wider_face_train_bbx_gt.txt"
    x_path_train = r"C:\Users\bobol\Desktop\CNN\faces\train"
    with open(y_path_train, "r") as f:
        train_lines = f.readlines()
    train_i = 0
    y_train = []
    x_train = []
    while train_i < 200:
        file_name = train_lines[train_i].strip()
        faces = train_lines[train_i + 1]
        face_count = int(faces)
        if face_count + 0:
            y_train.append(1)
        else:
            y_train.append(0)
        x_train.append(os.path.join(x_path_train,file_name))
        train_i += 2
        if face_count == 0:
            face_count+=1
        train_i += face_count

    # 验证集
    y_path_val = r"C:\Users\bobol\Desktop\CNN\faces\val\wider_face_val_bbx_gt.txt"
    x_path_val = r"C:\Users\bobol\Desktop\CNN\faces\val"
    with open(y_path_val, "r") as fv:
        val_lines = fv.readlines()
    val_i = 0
    y_val = []
    x_val = []
    while val_i < 200:
        file_name = val_lines[val_i].strip()
        file_name.replace("/","\\")
        faces = val_lines[val_i + 1]
        face_count = int(faces)
        if face_count + 0:
            y_val.append(1)
        else:
            y_val.append(0)
        x_val.append(os.path.join(x_path_val,file_name))
        val_i += 2
        if face_count == 0:
            face_count+=1
        
        val_i += face_count
    fr = {"a":x_train,"b":y_train}
    df = pandas.DataFrame(fr)
    print(df)
    return x_train, y_train, x_val, y_val

if __name__ == "__main__":
    print("ssss")
    get_datas()
