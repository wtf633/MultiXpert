import torch
import os
from tqdm import tqdm  # 导入 tqdm 库

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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


# 假设 get_projected_global_embedding 是一个计算 sim2 的函数
def calculate_sim2(img_path):
    return image_text_inference.image_inference_engine.get_projected_global_embedding(Path(img_path))


# 图像存储的路径
image_dir = 'data/0-clahe'  # 替换为实际的图像路径
save_path = 'data/0-clahe.pt'  # 所有 sim2 的保存路径

# 创建一个字典来存储所有 sim2 向量
all_sim2 = {}

# 获取文件列表
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# 遍历路径中的所有图像，使用 tqdm 包装以显示进度条
for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(image_dir, img_file)

    # 计算 sim2
    sim2 = calculate_sim2(img_path)

    # 使用图像的文件名（不带扩展名）作为字典的键
    all_sim2[Path(img_file).stem] = sim2

# 保存所有 sim2 到一个文件
torch.save(all_sim2, save_path)
print(f"All sim2 vectors saved at {save_path}")
