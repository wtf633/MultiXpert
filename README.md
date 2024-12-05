## MultiXpert: Multi-Modal Collaborative Enhanced Zero-Shot Diagnosis Model for Chest X-Ray Images

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

## Proposed method 

#### Our previous research: https://github.com/wtf633/Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases

This study introduces an innovative multi-modal collaborative enhancement framework, MultiXpert, designed to improve the perception and generalization performance of zero-shot diagnosis models for lung diseases. By optimizing visual features from medical images and refining textual features from report descriptions, the framework addresses critical challenges in zero-shot diagnosis.

<img src="https://github.com/wtf633/MultiXpert/blob/main/framework.png" alt="示例图片" width="800">

The overall structure of MultiXpert is illustrated in Figure, with the core components comprising the following modules: **Visual Semantic Enhancement:** (A) Spatial Domain Adaptive Correction module: Enhances medical images by reducing heterogeneity and restoring lesion details. (B) Positional Mask Prompt module: Extracts key visual features by guiding the visual encoder to focus on both global and local regions. **Textual Semantic Refinement:** (C) Description Refinement and Granularity Enhancement module: Improves disease description texts by reorganizing and refining semantic details, enabling the text encoder to extract core semantic features. **Feature Storage and Fusion:** (D) Feature Memory Matrix module: Stores key visual features extracted from the training set, serving as a repository for cross-modal comparisons. (E) Multi-Modal Similarity Comparison module: Fuses visual and textual features to compute the probability of disease occurrence through cross-modal alignment.

## Experimental datasets
#### seen Diseases: 
1) Zhanglab consists of 6,480 frontal chest X-rays, each labeled as "normal" or "pneumonia." The dataset is divided into a training set (1,349 normal and 3,883 pneumonia images) and a test set (234 normal and 390 pneumonia images). (https://data.mendeley.com/datasets/rscbjbr9sj/3)
2) CheXpert contains 224,316 chest X-rays from 65,240 patients collected at Stanford Hospital. The official validation set and test set include 200 and 500 images, respectively. Frontal PA images were evaluated on the official test set, covering 12 distinct abnormalities. (https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
3) RSNA from the Radiological Society of North America Pneumonia Detection Challenge is designed for binary classification tasks (pneumonia vs. normal). The training, validation, and test sets contain 25,184, 1,500, and 3,000 images, respectively. (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/)
4) VinDr-CXR provided by the Vingroup Big Data Research Institute, this dataset contains over 18,000 CXR scans collected from two major Vietnamese hospitals (2018–2020), covering 14 types of abnormalities. For evaluation, we selected 1,000 normal and 1,000 abnormal images as the test set. (https://vindr.ai/datasets)
#### Unseen Diseases: COVID-19
5) COVID-QED includes 33,920 chest X-rays, comprising 11,956 COVID-19 cases, 11,263 non-COVID infection cases, and 10,701 normal cases. We evaluated the COVID-19 segmentation data with a resolution of 256×256 pixels. (https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
6) COVID-19-CXD contains 21,165 chest X-rays, categorized as 3,616 COVID-19 positive, 6,012 lung opacity, 1,345 viral pneumonia, and 10,192 normal images. The dataset, with a resolution of 299×299 pixels, was developed collaboratively by doctors from Qatar University, Dhaka University, and Pakistan. (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
7) COVIDx comprises 84,818 chest X-rays from 45,342 patients. Created by the Vision and Image Processing Research Group at the University of Waterloo, it was evaluated using the COVIDx9B test dataset with a resolution of 1024×1024 pixels. (https://github.com/lindawangg/COVID-Net/tree/master)

## Implementation
#### Image Enhancement
- Run the code **preprocessing.py** to implement image augmentation. You can use it on public datasets or your own chest X-ray image data.
#### Positional Mask
- To do this, you need to replace the **interference_engine.py** file under the health_multimodal function package.
#### Text Enhancement
- Process the text in the order in which it is placed in the folder. Before doing this, you need to obtain the [MIMIC](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) electronic report dataset.
#### Feature Memory Matrix
- Select representative labeled data from the training set. Note that our image augmentation code needs to be used before use to maintain consistency with subsequent feature alignment between images.
#### inference
- Run the **inference.py** directly to infer your data!

## Acknowledgment
This work was supported by the National Natural Science Foundation of China (82371931), the Anhui Provincial Key Research and Development Project (2023s07020001). In addition, we sincerely thank the eight township hospitals for their collaboration and the two campuses of Hefei Cancer Hospital, Chinese Academy of Sciences, for their generous support in providing private data.

#### Here, we would like to express our special thanks to [Xplainer](https://github.com/ChantalMP/Xplainer/tree/master)[1] and [Style-Aware Radiology Report Generation](https://aclanthology.org/2023.findings-emnlp.977/)[2] for providing the code base.
[1] C. Pellegrini, M. Keicher, E. O¨ zsoy, P. Jiraskova, R. Braren, and N.Navab, “Xplainer: From x-ray observations to explainable zero-shot diagnosis,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2023: Springer, pp. 420-429. 

[2] Benjamin Yan, Ruochen Liu, David Kuo, Subathra Adithan, Eduardo Reis, Stephen Kwak, Vasantha Venugopal, Chloe O’Connell, Agustina Saenz, Pranav Rajpurkar, and Michael Moor. 2023. Style-Aware Radiology Report Generation with RadGraph and Few-Shot Prompting. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14676–14688.
