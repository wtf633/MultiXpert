import os
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from pathlib import Path
from typing import Tuple
from typing import List
from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.text import get_cxr_bert_inference
from health_multimodal.vlp import ImageTextInferenceEngine
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map

text_inference = get_cxr_bert_inference()
image_inference = get_biovil_resnet_inference()

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)


# 提取特征函数
def extract_features_from_path_by_class(base_path, feature_extraction_func):
    """
    从 0 和 1 类别子文件夹中提取特征
    :param base_path: 根路径
    :param feature_extraction_func: 提供图像路径并返回嵌入特征的自定义函数
    :return: 字典，包含类别 0 和类别 1 的特征及路径
    """
    feature_dict = {}
    for class_label in ['0', '1']:
        class_path = Path(base_path) / class_label
        image_paths = sorted(class_path.glob('*.jpg')) + sorted(class_path.glob('*.png')) + sorted(
            class_path.glob('*.jpeg'))
        features = []
        for image_path in image_paths:
            feature = feature_extraction_func(image_path)
            features.append(feature)
        feature_dict[class_label] = {
            'features': np.array(features),
            'paths': image_paths
        }
    return feature_dict


# 2. 自定义特征提取函数 (假设你已定义好特征提取器)
def custom_feature_extraction(image_path):
    """
    使用你的模型获取图像的特征
    :param image_path: 图像路径 (字符串)
    :return: 嵌入特征
    """
    image_path = Path(image_path)
    image_embedding = image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)
    if isinstance(image_embedding, torch.Tensor):
        image_embedding = image_embedding.cpu().numpy()
    return image_embedding


# 3. 特征提取
# 提供增强前（x）和增强后（y）的图像存放路径
path_original = "data/RSNA/before"
path_enhanced = "data/RSNA/after"

features_original = extract_features_from_path_by_class(path_original, custom_feature_extraction)
features_enhanced = extract_features_from_path_by_class(path_enhanced, custom_feature_extraction)


# 可视化函数
def visualize_with_tsne_by_class(features_dict1, features_dict2, labels1, labels2, save_path):
    """
    使用 t-SNE 可视化两个类别的特征分布，分别绘制增强前后两张子图
    :param features_dict1: 增强前特征字典
    :param features_dict2: 增强后特征字典
    :param labels1: 增强前标签
    :param labels2: 增强后标签
    :param save_path: 保存图像的路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, features_dict, label in zip(axes, [features_dict1, features_dict2], [labels1, labels2]):
        # 合并类别特征
        combined_features = np.vstack([features_dict['0']['features'], features_dict['1']['features']])
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(combined_features)

        # 绘制类别 0 的特征分布
        ax.scatter(
            reduced_features[:len(features_dict['0']['features']), 0],
            reduced_features[:len(features_dict['0']['features']), 1],
            label='Normal', alpha=0.7, marker='o'
        )

        # 绘制类别 1 的特征分布
        ax.scatter(
            reduced_features[len(features_dict['0']['features']):, 0],
            reduced_features[len(features_dict['0']['features']):, 1],
            label='Pneumonia', alpha=0.7, marker='x'
        )

        ax.legend()
        ax.set_title(label)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# 绘制并保存子图
visualize_with_tsne_by_class(
    features_original, features_enhanced,
    labels1="Original (Before Enhancement)",
    labels2="Preprocessing (After Enhancement)",
    save_path="tsne_comparison_hos.png"
)

print("t-SNE 可视化完成，图像已保存为 'tsne_comparison.png'")
