from pathlib import Path
from typing import List
import glob
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.text import get_cxr_bert_inference
from health_multimodal.vlp import ImageTextInferenceEngine

from pre_prompts import pres

from utils import cos_sim_to_prob, prob_to_log_prob, log_prob_to_prob


class InferenceModel():
    def __init__(self):
        self.text_inference = get_cxr_bert_inference()
        self.image_inference = get_biovil_resnet_inference()

        self.image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=self.image_inference,
            text_inference_engine=self.text_inference,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_text_inference.to(self.device)

        # caches for faster inference
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

        self.transform = self.image_inference.transform

    def get_similarity_score_from_raw_data(self, image_embedding, query_text: str) -> float:
        """Compute the cosine similarity score between an image and one or more strings.
        If multiple strings are passed, their embeddings are averaged before L2-normalization.
        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        assert not self.image_text_inference.image_inference_engine.model.training
        assert not self.image_text_inference.text_inference_engine.model.training
        if query_text in self.text_embedding_cache:
            text_embedding = self.text_embedding_cache[query_text]
        else:
            text_embedding = self.image_text_inference.text_inference_engine.get_embeddings_from_prompt([query_text],
                                                                                                        normalize=False)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding = F.normalize(text_embedding, dim=0, p=2)
            self.text_embedding_cache[query_text] = text_embedding

        cos_similarity = image_embedding @ text_embedding.t()

        return cos_similarity.item()

    def process_image(self, image):
        ''' same code as in image_text_inference.image_inference_engine.get_projected_global_embedding() but adapted to deal with image instances instead of path'''

        transformed_image = self.transform(image)
        projected_img_emb = self.image_inference.model.forward(transformed_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)
        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2
        return projected_img_emb[0]

    def get_descriptor_probs1(self, image_path: Path, descriptors: List[str], directions, weights):
        probs = {}
        negative_probs = {}

        image_embeddings = {
            dir: self.image_text_inference.image_inference_engine.get_projected_global_embedding1(image_path, dir) for
            dir in directions}
        pre = random.choice(pres)
        # 生成提示词并计算分数
        for desc in descriptors:
            prompts = f'{pre}There are {desc}'
            neg_prompts = f'{pre}There are no {desc}'

            weighted_image_embedding = sum(weights[i] * image_embeddings[dir] for i, dir in enumerate(directions))
            avg_score = self.get_similarity_score_from_raw_data(weighted_image_embedding, prompts)
            avg_neg_score = self.get_similarity_score_from_raw_data(weighted_image_embedding, neg_prompts)

            # 如果启用了负向提示词，计算负向概率
            pos_prob, neg_prob = torch.softmax(torch.tensor([avg_score, avg_neg_score]) / 0.5, dim=0)
            negative_probs[desc] = neg_prob

            probs[desc] = pos_prob

        return probs, negative_probs

    def get_probs(self, image_path: Path, selected_keys_normal, selected_keys_abnormal, all_sim_normal,
                  all_sim_abnormal, num_samples_normal, num_samples_abnormal):
        def compute_average_similarity(sim1, selected_keys, all_sim2, num_samples):
            total_similarity = 0.0
            for key in selected_keys:
                sim2 = all_sim2[key]
                total_similarity += (sim1 @ sim2.t()).item()  # 计算余弦相似度并转化为标量
            return total_similarity / num_samples

        sim1 = self.image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)

        # 计算NORMAL和PNEUMONIA文件夹的平均相似度
        average_similarity = compute_average_similarity(sim1, selected_keys_normal, all_sim_normal, num_samples_normal)
        neg_average_similarity = compute_average_similarity(sim1, selected_keys_abnormal, all_sim_abnormal,
                                                            num_samples_abnormal)

        similarity_tensor = torch.tensor([average_similarity, neg_average_similarity])

        # 计算softmax，温度系数为0.5
        probabilities = torch.softmax(similarity_tensor / 0.5, dim=0)
        return probabilities

    def get_all_descriptors(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc} indicating {disease}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    def get_all_descriptors_only_disease(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    def get_diseases_probs(self, disease_descriptors, pos_probs, negative_probs, prior_probs=None,
                           do_negative_prompting=True):
        disease_probs = {}
        disease_neg_probs = {}
        for disease, descriptors in disease_descriptors.items():
            desc_log_probs = []
            desc_neg_log_probs = []
            for desc in descriptors:
                desc = f"{desc} indicating {disease}"
                desc_log_probs.append(prob_to_log_prob(pos_probs[desc]))
                if do_negative_prompting:
                    desc_neg_log_probs.append(prob_to_log_prob(negative_probs[desc]))
            disease_log_prob = sum(sorted(desc_log_probs, reverse=True)) / len(desc_log_probs)

            if do_negative_prompting:
                disease_neg_log_prob = sum(desc_neg_log_probs) / len(desc_neg_log_probs)
            disease_probs[disease] = log_prob_to_prob(disease_log_prob)
            if do_negative_prompting:
                disease_neg_probs[disease] = log_prob_to_prob(disease_neg_log_prob)

        return disease_probs, disease_neg_probs

    # Threshold Based
    def get_predictions(self, disease_descriptors, threshold, disease_probs, keys):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'Normal':
                continue
            prob_vector[keys.index(disease)] = disease_probs[disease]
            if disease_probs[disease] > threshold:
                predicted_diseases.append(disease)

        if len(predicted_diseases) == 0:  # No finding rule based
            prob_vector[0] = 1.0 - max(prob_vector)
        else:
            prob_vector[0] = 1.0 - max(prob_vector)

        return predicted_diseases, prob_vector

    # Negative vs Positive Prompting
    def get_predictions_bin_prompting(self, disease_descriptors, disease_probs, negative_disease_probs, keys):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'Normal':
                continue
            pos_neg_scores = torch.tensor([disease_probs[disease], negative_disease_probs[disease]])
            prob_vector[keys.index(disease)] = pos_neg_scores[0]
            if torch.argmax(pos_neg_scores) == 0:  # Positive is More likely
                predicted_diseases.append(disease)

        if len(predicted_diseases) == 0:  # No finding rule based
            prob_vector[0] = 1.0 - max(prob_vector)
        else:
            prob_vector[0] = 1.0 - max(prob_vector)

        return predicted_diseases, prob_vector
