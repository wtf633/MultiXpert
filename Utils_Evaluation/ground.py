import os
from typing import List, Tuple
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

import torch
from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize inference engines
text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)
image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)

TypeBox = Tuple[float, float, float, float]


# Define the function for plotting phrase grounding
def plot_phrase_grounding(image_path: Path, text_prompt: str, bboxes: List[TypeBox]) -> Tuple[float, any]:
    similarity_map = image_text_inference.get_similarity_map_from_raw_data(
        image_path=image_path,
        query_text=text_prompt,
        interpolation="bilinear",
    )
    fig = plot_phrase_grounding_similarity_map(
        image_path=image_path,
        similarity_map=similarity_map,
        bboxes=bboxes
    )
    score = image_text_inference.get_similarity_score_from_raw_data(image_path=image_path, query_text=text_prompt)
    return score, fig


# Define directories and bounding boxes
input_dir = Path("data/RSNA/images-right")  # Directory containing input images
output_dir = Path("data/RSNA/images-clahe_mask-right basilar")  # Directory to save results
output_dir.mkdir(parents=True, exist_ok=True)
bboxes = []

# List of text prompts to iterate over
text_prompts = [
    "",
    "right basilar Ground-glass opacities seen",
    "right basilar Vascular thickening seen",
    "right basilar Septal thickening seen",
    "right basilar Small patchy shadows seen",
    "right basilar Interstitial changes seen",
    "right basilar Reticular pattern seen",
    "right basilar Interstitial involvement seen",
    "right basilar Pleural effusion seen",
    "right basilar Consolidation seen"
]

# Process each image in the input directory
for image_path in input_dir.iterdir():
    if image_path.is_file():
        # Create a unique directory for the current image in the output path
        image_output_dir = output_dir / image_path.stem
        for text_prompt in text_prompts:
            score, figure = plot_phrase_grounding(image_path, text_prompt, bboxes)
            # Save figure if the similarity score is greater than 0.5
            # if score < -0.6 or (0.1 > score > -0.2):
            if score < 1.0:
                if score > 0.55:
                    print(f"相似度得分 for '{image_path.stem}-{text_prompt}': {score:.3f}")
                if os.path.exists(image_output_dir):
                    pass
                else:
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                save_path = image_output_dir / f"{image_path.stem}-{text_prompt.replace(' ', '_')}-{score:.3f}.png"
                figure.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(figure)  # 尝试关闭 figure 对象
            plt.close('all')  # 进一步确保释放所有图像资源
