from math import log, exp

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve
import pandas as pd
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def cos_sim_to_prob(sim):
    return (sim + 1) / 2  # linear transformation to 0 and 1


def log_prob_to_prob(log_prob):
    return exp(log_prob)


def prob_to_log_prob(prob):
    return log(prob)


def calculate_total(y_pred, y_true):
    # 将标签转换为二进制格式
    y_true_binary = np.argmax(y_true, axis=1)
    y_pred_binary = np.argmax(y_pred, axis=1)

    # 计算 AUC（需要提供预测的概率）
    auc = roc_auc_score(y_true, y_pred)
    # 计算 ACC
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    # 计算 F1 分数
    f1 = f1_score(y_true_binary, y_pred_binary)
    # 计算 AP（average precision）
    ap = average_precision_score(y_true, y_pred)
    # print(f"ACC: {accuracy*100:.3f}")
    # print(f"F1: {f1*100:.3f}")
    # print(f"AUC: {auc*100:.3f}")
    # print(f"AP: {ap*100:.3f}")
    # print("-"*50)
    return accuracy, f1, auc, ap
