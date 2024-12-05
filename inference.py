import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chestxray14 import ZhangDataset, CheDataset, VinDataset, RSNADataset, COVIDDataset, COVIDXDataset, COVID19Dataset, MultiDataset, RSNAGoodDataset
from descriptors import disease_descriptors_Pneumonia, disease_descriptors_Abnormal, disease_descriptors_COVID19
from model import InferenceModel
from utils import calculate_auroc, calculate_total, calculate_total_m, calculate_total_ml

torch.multiprocessing.set_sharing_strategy('file_system')


def inference(dataset, pro_num, directions, weights):
    print(dataset)
    accuracies = []
    f1_scores = []
    auc_scores = []
    ap_scores = []
    normal_folder = ""
    pneumonia_folder = ""
    disease_descriptors = ""
    dataset = RSNADataset(f'data/RSNA/file_names.csv')
    normal_folder = "/home/image023/data/PXplainer/Sim-Fig/sim-extra/RSNA-good/RSNA-clahe-sim2-normal.pt"
    pneumonia_folder = "/home/image023/data/PXplainer/Sim-Fig/sim-extra/RSNA-good/RSNA-clahe-sim2-pneumonia.pt"
    disease_descriptors = disease_descriptors_Pneumonia

    all_sim_normal = torch.load(normal_folder)
    all_sim_abnormal = torch.load(pneumonia_folder)

    for _ in range(1):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
        inference_model = InferenceModel()
        all_descriptors = inference_model.get_all_descriptors(disease_descriptors)

        num_samples_normal = int(len(all_sim_normal) * pro_num)
        num_samples_abnormal = int(len(all_sim_abnormal) * pro_num)
        selected_keys_normal = random.sample(list(all_sim_normal.keys()), num_samples_normal)
        selected_keys_abnormal = random.sample(list(all_sim_abnormal.keys()), num_samples_abnormal)

        all_labels = []
        all_probs_neg = []
        tag1 = []
        tag2 = []
        paths = []
        for batch in tqdm(dataloader):
            batch = batch[0]
            image_path, labels, keys = batch
            image_path = Path(image_path)
            paths.append(image_path)
            prob_vector_neg_prompt1 = inference_model.get_probs(image_path, selected_keys_normal,
                                                                selected_keys_abnormal, all_sim_normal,
                                                                all_sim_abnormal, num_samples_normal,
                                                                num_samples_abnormal)
            probs, negative_probs = inference_model.get_descriptor_probs1(image_path, descriptors=all_descriptors, directions=directions, weights=weights)
            # probs, negative_probs = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
            disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors=disease_descriptors,
                                                                                       pos_probs=probs,
                                                                                       negative_probs=negative_probs)
            predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
                disease_descriptors,
                disease_probs=disease_probs,
                negative_disease_probs=negative_disease_probs,
                keys=keys)
            all_labels.append(labels)
            all_probs_neg.append((prob_vector_neg_prompt + prob_vector_neg_prompt1) / 2)

        all_labels = torch.stack(all_labels)
        all_probs_neg = torch.stack(all_probs_neg)

        existing_mask = sum(all_labels, 0) > 0
        all_labels_clean = all_labels[:, existing_mask]
        all_probs_neg_clean = all_probs_neg[:, existing_mask]

        accuracy, f1, auc, ap = calculate_total(all_probs_neg_clean, all_labels_clean)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        auc_scores.append(auc)
        ap_scores.append(ap)


    accuracy_mean, accuracy_std = np.mean(accuracies), np.std(accuracies)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    ap_mean, ap_std = np.mean(ap_scores), np.std(ap_scores)

    print(f"Accuracy: {accuracy_mean * 100:.1f}  {accuracy_std * 100:.1f}")
    print(f"F1 Score: {f1_mean * 100:.1f}  {f1_std * 100:.1f}")
    print(f"AUC: {auc_mean * 100:.1f}  {auc_std * 100:.1f}")
    print(f"AP: {ap_mean * 100:.1f}  {ap_std * 100:.1f}")


if __name__ == '__main__':
    directions = ["left", "right", "top", "bottom", "right top", "right bottom", "left top", "left bottom", "all"]
    weights_RSNA = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]  # RSNA
    inference(dataset="RSNA ", pro_num=0.5, directions=directions)
