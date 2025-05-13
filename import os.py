import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ----------------------
# 1. 数据准备
# ----------------------


class CatDogDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        """
        自定义数据集类
        :param img_dir: 图片目录路径
        :param json_path: 包含标签的json文件路径
        :param transform: 数据增强/预处理
        """
        self.img_dir = img_dir
        self.transform = transform

        # 加载标签数据
        with open(json_path) as f:
            self.labels = json.load(f)  # 假设格式如 {"img_001.jpg": "cat", ...}

        # 获取所有图片文件名
        self.img_files = [
            f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 加载图片
        image = Image.open(img_path).convert("RGB")  # 确保转为RGB格式

        # 获取标签并转为数字（猫:0，狗:1）
        label_str = self.labels.get(img_name, "unknown")
        label = 0 if label_str.lower() == "cat" else 1

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据增强和预处理
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # 统一尺寸（如果图片尺寸不一致）
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转（-15到15度）
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(  # 标准化（ImageNet均值方差）
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# 创建完整数据集
dataset = CatDogDataset(
    img_dir="./your_images_folder",  # 替换为你的图片目录
    json_path="./labels.json",  # 替换为你的json文件路径
    transform=transform,
)

# 划分训练集和验证集（8:2比例）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# ----------------------
# 2. 定义模型
# ----------------------


class CatDogClassifier(nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # 输入: 3x128x128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 32x64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 64x32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 128x16x16
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(512, 1),  # 二分类输出
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x


# 初始化模型
model = CatDogClassifier()

# ----------------------
# 3. 训练配置
# ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率衰减

# ----------------------
# 4. 训练循环
# ----------------------

num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # 调整形状匹配模型输出

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # 计算准确率
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # 计算统计量
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    # 打印进度
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # 调整学习率
    scheduler.step()

print("训练完成！最佳验证准确率: {:.2f}%".format(best_val_acc * 100))
