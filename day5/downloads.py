import os
import cv2
from google_images_download import google_images_download

pathss="0-nof"

def download_images(query, num_images, output_dir):
    response = google_images_download.googleimagesdownload()
    arguments = {
        "keywords": query,
        "limit": num_images,
        "output_directory": output_dir,
        "image_directory": query,
        "no_download": False,
        "format": "jpg"
    }
    paths = response.download(arguments)
    return paths

def generate_labels(image_dir, label_file):
    with open(label_file, 'w') as f:
        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                f.write(f"{pathss}/{filename}\n0 0 0 0 0 0 0 0 0 0\n")

    
# 设置参数
query = "nature scenery"  # 搜索关键词
num_images = 10  # 下载图片数量
output_dir = "./negative_samples"  # 图片保存目录
label_file = "./negative_samples/labels.txt"  # 标签文件路径

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 下载图片
download_images(query, num_images, output_dir)

# 生成标签文件
image_dir = os.path.join(output_dir, query)
generate_labels(image_dir, label_file)

print(f"下载完成！标签文件已保存至 {label_file}")