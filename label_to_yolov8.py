import cv2
import os
import math
import numpy as np
import fnmatch
from tqdm import *


# 去除边缘白边
def remove_frame(img_path):
    # 读取图像
    img_copy = cv2.imread(img_path).copy()
    img = cv2.imread(img_path, 0)  # 读取为灰度图像
    # 二值化
    thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)[1]
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓
    cnt = max(contours, key=cv2.contourArea)
    # 获取外接矩形
    x, y, w, h = cv2.boundingRect(cnt)
    # 裁剪图像
    cropped_img = img_copy[y + 1:y + h, x:x + w]
    return cropped_img


image_paths = []
for root, dirs, files in os.walk(r"D:\Sheng\sheng\stars\STARS\data"):
    for file in files:
        if fnmatch.fnmatch(file.lower(), '*.jpg') or fnmatch.fnmatch(file.lower(), '*.png') or fnmatch.fnmatch(
                file.lower(), '*.jpeg'):
            image_paths.append(os.path.join(root, file))

output_dir = r"D:\Sheng\sheng\stars\test\yolov8\split_images\trainset"

for image_path in tqdm(image_paths):
    image_name = image_path.split("\\")
    image_name = image_name[-1].split(".")
    output_dir_images = f"{output_dir}\{image_name[-2]}\images"
    output_dir_labels = f"{output_dir}\{image_name[-2]}\labels"

    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # 读取图片
    image = cv2.imread(image_path)
    image = remove_frame(image_path)
    height, width = image.shape[:2]  # 获取图像的高度和宽度

    # 分割块大小
    block_size = 512  # 假设每个块为512x512
    rows = math.ceil(height / block_size)
    cols = math.ceil(width / block_size)

    # 遍历每个块
    for row in range(rows):
        for col in range(cols):
            # 计算每个小块的起始坐标
            x_start = col * block_size
            y_start = row * block_size
            if x_start + block_size <= width:
                x_end = x_start + block_size
            else:
                x_end = x_start + ((x_start + block_size) % width)
            if y_start + block_size <= height:
                y_end = y_start + block_size
            else:
                y_end = y_start + ((y_start + block_size) % height)

            # 提取小块
            block = image[y_start:y_end, x_start:x_end]
            # 小块路径
            block_image_path = os.path.join(output_dir_images,
                                            f"{image_name[-2]}_{row}_{col}.jpeg")
            block_txt_path = os.path.join(output_dir_labels,
                                          f"{image_name[-2]}_{row}_{col}.txt")

            # 保存小块
            cv2.imwrite(block_image_path, block)

            # 转换为灰度图并进行二值化
            gray_image = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
            _, binaryzation = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
            binaryzationctr_base, Bghier = cv2.findContours(binaryzation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # 打开标签文件
            with open(block_txt_path, 'w') as f:
                for cnt in binaryzationctr_base:
                    M = cv2.moments(cnt)
                    area = cv2.contourArea(cnt)
                    # 筛选满足条件的轮廓
                    if M["m00"] != 0 and area < 40000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 计算YOLO格式的中心点和宽高（归一化到0-1）
                        x_center = (x + w / 2) / block.shape[1]
                        y_center = (y + h / 2) / block.shape[0]
                        norm_w = w / block.shape[1]
                        norm_h = h / block.shape[0]
                        class_id = 0  # 假设所有区域的class_id为0
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    with open(os.path.join(output_dir_labels, 'classes.txt'), 'w') as f:
        f.write("0")
