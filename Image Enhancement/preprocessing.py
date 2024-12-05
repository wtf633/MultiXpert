import cv2 as cv
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image

image_dir = "data/RSNA/images"
save_dir = "data/RSNA/images-clahe"

patient_list = os.listdir(image_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
patient_list = os.listdir(image_dir)


def classify_image(mu, sigma):
    D = 4.0 * sigma  # Difference function
    r = 3.0
    # print("D: ", D)
    contrast_class = "low_contrast" if D <= 1 / r else "high_contrast"
    # Determine if the image is bright or dim
    brightness_class = "bright" if mu >= 0.5 else "dim"
    return contrast_class, brightness_class


def adjust_gamma(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv.LUT(image, table)


for patient in tqdm(patient_list):
    imageName = os.path.join(image_dir, patient)
    writeName = os.path.join(save_dir, patient)
    img = Image.open(imageName)
    image = cv.imread(imageName, cv.IMREAD_GRAYSCALE)

    # image = cv2.resize(image, (512, 512))

    sigma = np.std(image / 255.0)
    mu = np.mean(image / 255.0)
    contrast_class, brightness_class = classify_image(mu, sigma)

    if img.size[0] < 500:
        width_divided = 1
    else:
        width_divided = img.size[0] // 500
    if img.size[1] < 500:
        height_divided = 1
    else:
        height_divided = img.size[1] // 500

    if contrast_class == 'low_contrast':
        clahe = cv.createCLAHE(clipLimit=1 + 4.0 * sigma, tileGridSize=(width_divided, height_divided))
        image_clahe = clahe.apply(image)
    else:
        clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(width_divided, height_divided))
        image_clahe = clahe.apply(image)

    if brightness_class == 'bright':
        adjusted_image = adjust_gamma(image, gamma=0.8)
    else:
        adjusted_image = adjust_gamma(image, gamma=1.2)


    extra_sharpen_kernel = np.array([[1, -2, 1],
                                     [-2, 5, -2],
                                     [1, -2, 1]], np.float32)
    dst_extra_sharpen = cv2.filter2D(image_clahe, -1, extra_sharpen_kernel)

    edge_enhance_sharpen_kernel = np.array([[-1, -1, -1, -1, -1],
                                            [-1, 2, 2, 2, -1],
                                            [-1, 2, 16, 2, -1],
                                            [-1, 2, 2, 2, -1],
                                            [-1, -1, -1, -1, -1]]) / 16.0

    dst_edge_enhance = cv2.filter2D(image_clahe, -1, edge_enhance_sharpen_kernel)

    # 将两个处理后的图像合并
    # alpha, beta, gamma 是合并时的参数，可以根据需要调整
    # alpha 是 dst_extra_sharpen 的权重
    # beta 是 dst_edge_enhance 的权重
    # gamma 是一个添加到所有像素上的标量值，通常设为 0
    alpha = 0.5
    beta = 0.5
    g = 0

    combined_dst = cv2.addWeighted(dst_extra_sharpen, alpha, dst_edge_enhance, beta, g)
    combined_dst1 = cv2.addWeighted(combined_dst, alpha, adjusted_image, beta, g)

    cv.imwrite(writeName, combined_dst1)

