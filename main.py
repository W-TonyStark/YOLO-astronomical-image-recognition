import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import math
from tqdm import *
from noise_reduction import denoised
import point_match

image_base_path = r"D:\Sheng\sheng\stars\test\F001_2409+004_20240708\reference\F001_2409+004_40010230_110_0030000_20240708004120_00040_00001.jpeg"
# image_compare_path = r"D:\Sheng\sheng\stars\test\F001_2409+004_20240708\F001_2409+004_40010330_110_0030000_20240708005420_00040_00001.jpeg"
# 阈值
distance_threshold = 200.0
angle_threshold = 180
combined_threshold = 8000
area_threshold = 10
anomalous_points_threshold = 1.2

# 亮点面积大小范围
area_min = 3
area_max = 40000

# denoised_base = denoised(image_base_path)
# denoised_compare = denoised(image_compare_path)
# 设置特征点选取数量
kn = 7


def get_features_base(image):
    global binaryzationctr_base, image_with_contours_base, centroids_base
    # image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryzation = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    binaryzationctr_base, Bghier = cv2.findContours(binaryzation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image_with_contours_base = image.copy()  # 创建原图的副本
    # 找点
    centroids_base = find_point(binaryzationctr_base)
    # 获取特征值
    features = calculate_distance_angle(centroids_base, binaryzationctr_base)
    return features


def get_features_compare(image):
    global binaryzationctr_compare, image_with_contours_compare, centroids_compare
    # image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binaryzation = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    binaryzationctr_compare, Bghier = cv2.findContours(binaryzation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    image_with_contours_compare = image.copy()  # 创建原图的副本
    # 找点
    centroids_compare = find_point(binaryzationctr_compare)
    # 获取特征值
    features = calculate_distance_angle(centroids_compare, binaryzationctr_compare)
    return features


def find_point(binaryzationctr):
    # 创建一个列表存储质心坐标、亮点面积
    centroids = []
    num = 0
    # 遍历每个轮廓，计算质心，并给每个点标号
    for cnt in binaryzationctr:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if M["m00"] != 0 and area_min < area < area_max:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY, area, num))
            num += 1
    return centroids


# 定义函数计算角度
def angle(x1, y1, x2, y2):
    ang = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if abs(ang) >= 180:
        ang = 360 - abs(ang)
    return ang


def calculate_distance_angle(centroids, binaryzationctr):
    # 查找每个点的最近kn-1个点，计算距离和角度
    features = []
    # 提取所有点的(x,y)坐标
    cords = [(x, y) for x, y, _, _ in find_point(binaryzationctr)]
    # 构建 KDTree
    tree = KDTree(cords)
    for (x, y, a, i) in centroids:
        # 查找最近的kn个点（包括自己）
        dists, indices = tree.query((x, y), k=kn)
        # 去掉第一个点（自己）
        dists = dists[1:]
        indices = indices[1:]
        # 计算距离和角度
        feature = []
        for dist, idx in zip(dists, indices):
            x2, y2 = cords[idx]
            ang = angle(x, y, x2, y2)
            feature.append((dist, ang))
        feature.append(a)
        feature.append(i)
        features.append(feature)
    return features


# 进行图片对齐，消除图片轻微位移影响，并消除由于亮度造成的亮点面积大小轻微变化的影响
def align_images(centroids_base, centroids_compare, denoised_base, denoised_compare):
    sorted_lists_base = sorted(centroids_base, key=lambda x: x[2], reverse=True)[:2]
    sorted_lists_compare = sorted(centroids_compare, key=lambda x: x[2], reverse=True)[:2]
    area_move = 0
    for i in range(len(sorted_lists_base)):
        area_move += sorted_lists_base[i][2] / sorted_lists_compare[i][2]
    area_move = area_move / len(sorted_lists_base)
    # 使用ORB特征点检测
    orb = cv2.ORB_create()
    # 检测特征点和计算描述子
    kp1, des1 = orb.detectAndCompute(denoised_base, None)
    kp2, des2 = orb.detectAndCompute(denoised_compare, None)
    # 使用BFMatcher匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    # 获取匹配点对
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 使用RANSAC估计仿射变换矩阵
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2)
    # 提取平移量x_move, y_move
    x_move, y_move = M[0, 2], M[1, 2]
    return x_move, y_move, area_move


# 寻找异常运动点
def compute_movement_vectors(coord_list, image_num):
    # 计算连续坐标之间的运动矢量
    return [(coord_list[i + 1][0] - coord_list[i][0], coord_list[i + 1][1] - coord_list[i][1]) for i in
            range(image_num - 1)]


def detect_anomalies(data):
    all_vectors = []
    movement_vectors = {}

    # 计算每个点的运动矢量
    for key, coords in data.items():
        vectors = compute_movement_vectors(coords, len(coords))
        movement_vectors[key] = vectors
        all_vectors.extend(vectors)

    # 计算所有运动矢量的平均值和标准偏差
    all_vectors_np = np.array(all_vectors)
    mean_vector = np.mean(all_vectors_np, axis=0)
    std_vector = np.std(all_vectors_np, axis=0)

    anomalies = []

    # 通过将每个点的向量与全局均值进行比较来识别异常值
    for key, vectors in movement_vectors.items():
        deviations = [np.linalg.norm(np.array(vec) - mean_vector) for vec in vectors]
        avg_deviation = np.mean(deviations)

        # 如果平均偏差高于某个阈值，则将点标记为异常
        if avg_deviation > anomalous_points_threshold * np.linalg.norm(std_vector):
            anomalies.append(key)

    return anomalies


features_base = get_features_base(cv2.imread(image_base_path))
# 以base图为基准的所有图匹配上的点的坐标字典
match_points_dic = {}

base_num = 0
for cnt in binaryzationctr_base:
    M = cv2.moments(cnt)
    area = cv2.contourArea(cnt)
    if M["m00"] != 0 and area_min < area < area_max:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # 在图像上绘制质心和轮廓
        cv2.drawContours(image_with_contours_base, [cnt], -1, (0, 255, 0), -1)  # 绘制所有轮廓，绿色，线宽2
        cv2.circle(image_with_contours_base, (cX, cY), 1, (255, 0, 0), -1)
        # 绘制编号
        cv2.putText(image_with_contours_base, str(base_num), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        # 形成初始base图星点的字典
        match_points_dic[base_num] = [(cX, cY)]
        base_num += 1
image_with_contour_base = cv2.cvtColor(image_with_contours_base, cv2.COLOR_BGR2RGB)

image_compare_paths = []
for file in os.listdir(r"D:\Sheng\sheng\stars\test\F001_2409+004_20240708"):
    if file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(r"D:\Sheng\sheng\stars\test\F001_2409+004_20240708", file)
        image_compare_paths.append(image_path)
for image_compare_path in tqdm(image_compare_paths):
    print(image_compare_path)
    features_compare = get_features_compare(cv2.imread(image_compare_path))
    x_move, y_move, area_move = align_images(centroids_base, centroids_compare, cv2.imread(image_base_path),
                                             cv2.imread(image_compare_path))
    print("x_move:", x_move, "  y_move:", y_move, "  area_move:", area_move)
    # 提取compare所有点的(x,y)坐标
    compare_point = find_point(binaryzationctr_compare)
    cords_compare = [(x - x_move, y - y_move) for x, y, _, _ in compare_point]
    # 构建 KDTree
    tree = KDTree(cords_compare)
    # 设置一个compare图的点是否匹配的状态变量
    state_compare = np.zeros(len(cords_compare))
    # 统计共匹配了多少个点
    matched_num = 0
    # 在compare上找对应的点并作图
    for feature in tqdm(features_base):
        # 获取base图中点的ID
        id = feature[kn]
        feature = feature[:kn]
        x_base, y_base = centroids_base[id][0], centroids_base[id][1]
        # 初始化
        valid_indices = []
        valid_dists = []
        # 需要找到2个最近且状态为0的点
        k = 2
        query_count = 0
        while len(valid_indices) < k:
            # 查找最近的k个点
            dists, indices = tree.query((x_base, y_base), k=k + query_count)
            # 检查并收集状态为0的点
            for i, idx in enumerate(indices):
                if state_compare[idx] == 0:
                    valid_indices.append(idx)
                    valid_dists.append(dists[i])
                # 如果已经找到了2个有效点，就停止查找
                if len(valid_indices) == k:
                    break
            query_count += 1  # 每次增加查找范围，继续寻找
        nearest_points_features = [features_compare[i] for i in valid_indices]
        score = []
        id_feature = []
        for nearest_features in nearest_points_features:
            result = False
            # 只取特征值，抛去ID列
            candidate_features = nearest_features[:kn]
            candidate_features[kn - 1] *= area_move

            # 对compare点的kn-1个特征值进行筛选，避免因为错误的框亮点造成的误判
            feature, candidate_features = point_match.match_feature(feature, candidate_features, dist=2, angle=3,
                                                                    match_limit=3)
            # 查看匹配情况
            result = point_match.match_point(feature, candidate_features, distance_threshold, angle_threshold,
                                             area_threshold, combined_threshold)
            if result:
                score.append((point_match.get_score(feature, candidate_features), nearest_features[kn]))
        if id in match_points_dic:
            if not score:
                # 从字典中去除未匹配上的点
                match_points_dic.pop(id)
            else:
                min_element = min(score, key=lambda x: x[0])
                # 获取最匹配的点的序号
                best_match_point = min_element[1]
                state_compare[best_match_point] = 1
                x, y = compare_point[best_match_point][0], compare_point[best_match_point][1]
                # 将compare中匹配的点加入字典中记录
                match_points_dic[id].append((x, y))
                # 在图像上绘制质心和轮廓
                cv2.circle(image_with_contours_compare, (x, y), 1, (255, 0, 0), -1)
                matched_num += 1

                # 绘制编号
                cv2.putText(image_with_contours_compare, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255),
                            1)
    print(len(match_points_dic))

# 找轨迹异常点
anomalous_points = detect_anomalies(match_points_dic)
print(f"异常的点: {anomalous_points}")

image_with_contour_compare = cv2.cvtColor(image_with_contours_compare, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 在第一个子图中显示第一张图像
axes[0].imshow(image_with_contour_base, cmap='gray')
axes[0].set_title('Base Image')

# 在第二个子图中显示第二张图像
axes[1].imshow(image_with_contour_compare, cmap='gray')
axes[1].set_title('Compare Image')

print('matched points: ' + str(matched_num))
print("image saving...")
# plt.savefig('point_match_all.png', dpi=1500)
# plt.show()
print('image saved!')

# # 按照第三个元素也就是面积进行降序排序,找面积最大的十个框选区域
# sorted_lists = sorted(centroids, key=lambda x: x[2], reverse=True)
# top = sorted_lists[:10]
#
# top1 = [(1768, 441, 1468.5), (627, 2200, 1347.0), (3000, 2095, 1215.5), (3933, 4305, 1148.0), (2062, 4404, 1106.0),
#         (4961, 4587, 959.0), (3027, 3568, 951.5), (120, 2259, 935.0), (3992, 3938, 839.0), (776, 3238, 828.5)]
# top2 = [(1768, 441, 2434.0), (3001, 2104, 1715.0), (627, 2209, 1697.5), (3934, 4314, 1415.5), (2062, 4413, 1398.5),
#         (3027, 3577, 1131.5), (4962, 4597, 1111.0), (120, 2268, 1081.5), (3993, 3947, 955.0), (776, 3247, 941.5)]
#
# top1_sorted = sorted(top1, key=lambda x: x[0], reverse=True)
# top2_sorted = sorted(top2, key=lambda x: x[0], reverse=True)
#
# # 计算两图x,y以及面积s三者变化的值
# x, y, s = 0, 0, 0
# for i in range(len(top1_sorted)):
#     x += top1_sorted[i][0] - top2_sorted[i][0]
#     y += top1_sorted[i][1] - top2_sorted[i][1]
#     s += (top1_sorted[i][2] - top2_sorted[i][2])/top1_sorted[i][2]
# x_move = x/len(top1_sorted)
# y_move = y/len(top1_sorted)
# s_move_per = s/len(top1_sorted)
#
# # 给基准图每个点一个序号
# for index, item in enumerate(centroids):
#     centroids[index] = (*item, index + 1)
#
# print(top1_sorted)
# print(top2_sorted)
# print(x_move)
# print(y_move)
# print(s_move_per)
