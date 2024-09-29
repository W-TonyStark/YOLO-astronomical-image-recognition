import numpy as np


def match_point(target_features, candidate_features, distance_threshold, angle_threshold, area_threshold,
                combined_threshold):
    if not target_features or not candidate_features:
        return False
    # 匹配候选点
    independent_matched = match_points_independent(target_features, candidate_features, distance_threshold,
                                                   angle_threshold, area_threshold)
    combined_matched = match_points_combined(target_features, candidate_features, combined_threshold,
                                             area_threshold)  # 设置合适的综合阈值
    if independent_matched and combined_matched:
        return True
    return False


def get_score(target_features, candidate_features):
    # 将距离和角度对转换为向量
    target_vector = np.array(target_features[:len(target_features) - 1]).flatten()
    candidate_vector = np.array(candidate_features[:len(candidate_features) - 1]).flatten()
    # 计算两者的欧氏距离
    euclidean_distance = np.linalg.norm(target_vector - candidate_vector, ord=1)
    # 返回得分，值越低则越匹配
    return euclidean_distance


# 独立特征匹配函数
def match_points_independent(target_features, candidate_features, distance_threshold, angle_threshold, area_threshold):
    for (target_dist, target_angle), (cand_dist, cand_angle) in zip(target_features[:len(target_features) - 1],
                                                                    candidate_features[:len(candidate_features) - 1]):
        # 检查距离和角度差异是否在阈值范围内
        if abs(abs(target_dist) - abs(cand_dist)) > distance_threshold or abs(
                abs(target_angle) - abs(cand_angle)) > angle_threshold or (
                target_features[len(target_features) - 1] - candidate_features[len(candidate_features) - 1]) / \
                target_features[len(target_features) - 1] > area_threshold:
            return False
    return True


# 综合特征匹配函数
def match_points_combined(target_features, candidate_features, combined_threshold, area_threshold):
    # 将距离和角度对转换为向量，先忽略最后单独的大小面积数据列
    target_vector = np.array(target_features[:len(target_features) - 1]).flatten()
    candidate_vector = np.array(candidate_features[:len(candidate_features) - 1]).flatten()
    # 计算两者的欧氏距离
    euclidean_distance = np.linalg.norm(target_vector - candidate_vector, ord=1)
    # 计算两者亮点面积大小差异
    area_diff = (target_features[len(target_features) - 1] - candidate_features[len(candidate_features) - 1]) / \
                target_features[len(target_features) - 1]
    # 如果欧氏距离与面积差异均小于阈值，则认为匹配
    return euclidean_distance < combined_threshold and area_diff < area_threshold


def match_feature(base, compare, dist, angle, match_limit):
    matched_base = []
    matched_compare = []

    for i in range(len(base) - 1):
        base1, base2 = base[i]
        for j in range(len(compare) - 1):
            compare1, compare2 = compare[j]

            # 根据给定的阈值比较元素
            if abs(base1 - compare1) <= dist and abs(base2 - compare2) <= angle:
                matched_base.append(base[i])
                matched_compare.append(compare[j])
                break

    # 检查匹配次数是否超过限制
    if len(matched_base) >= match_limit:
        # 包含最后一个元素（面积元素）
        matched_base.append(base[-1])
        matched_compare.append(compare[-1])
        return matched_base, matched_compare
    else:
        return [], []

# # 点的特征列表
# target_features = [(18.601075237738275, 126.2538377374448),
#                    (29.274562336608895, 97.85331330197823),
#                    (40.26164427839479, 14.381394591090602),
#                    (52.3450093132096, 46.548157698977974),
#                    (62.0, 0.0),
#                    (64.35060217278468, 57.050784883409555)]
#
# # 候选点的特征列表
# candidate_features = [(19.601075237738275, 126.2538377374448),
#                       (32.274562336608895, 97.85331330197823),
#                       (40.26164427839479, 14.381394591090602),
#                       (52.3450093132096, 46.548157698977974),
#                       (62.0, 0.0),
#                       (64.35060217278468, 57.050784883409555)]  # 另一个图片中的6个(距离, 角度)对
#
# # 距离和角度的阈值
# distance_threshold = 5.0  # 根据需要调整
# angle_threshold = 5.0  # 根据需要调整
#
#
# result = match_point(target_features, candidate_features, distance_threshold, angle_threshold)
# print(result)
# print(get_score(target_features, candidate_features))
