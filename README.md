<div align="center">

# 🌿 PlantDoctor AI

### Plant Leaf Disease Classification Using Traditional Machine Learning

**15 classes · 3 crop types · 1,632 engineered features · 350 selected features · LightGBM + Streamlit**

[GitHub Repository](https://github.com/nada-ali-13/PlantDoctor-AI) · [Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

</div>

---

## Executive Summary

**PlantDoctor AI** is an image-based plant leaf disease classification system for **tomato, potato, and bell pepper**. It combines conventional computer-vision feature engineering with supervised machine learning and makes predictions through a Streamlit web application. Unlike deep-learning-based approaches, this project uses **HSV color histograms, RGB statistics, Local Binary Patterns (LBP), and Histogram of Oriented Gradients (HOG)** with a **LightGBM** classifier.

The final model was evaluated on a **held-out test set of 3,094 images**, achieving **95.1842% accuracy** and **93.8936% macro F1**. The result reflects performance on this PlantVillage-based dataset; it should not be interpreted as guaranteed performance on arbitrary field photographs.

## 1. Problem Statement and Objectives

Plant leaf diseases can be difficult to distinguish visually, particularly when different diseases produce similar symptoms or when the same disease affects multiple crop species. The objective is to develop a practical machine-learning prototype that:

- Classifies a leaf image into one of **15 crop-and-condition classes**.
- Uses **traditional machine learning**, rather than a deep neural network.
- Compares baseline and more advanced models using validation data.
- Preserves an independent test set for final assessment.
- Exposes the selected model through an accessible **Streamlit** interface.

The system is an **educational decision-support prototype**, not a substitute for professional plant pathology assessment.

## 2. Dataset

- **Source:** [PlantDisease / PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Selected source directory:** `plantvillage/PlantVillage` (one dataset copy, avoiding parallel duplicate folders)
- **Initial image/feature rows:** **20,638**
- **Identical feature rows removed before splitting:** **14**
- **Unique feature rows used for the split:** **20,624**
- **Crops:** Tomato, potato, and bell pepper
- **Target classes:** **15** combined crop/health-condition categories

### Data split

| Subset | Images | Approx. share |
|:--|--:|--:|
| Training | 14,436 | 70% |
| Validation | 3,094 | 15% |
| Independent test | 3,094 | 15% |
| **Total** | **20,624** | **100%** |

The original project used **stratified random splits** with `random_state=42`. The split indices refer to the **original feature matrix rows**, and the aligned metadata file preserves the corresponding class labels. The held-out test subset was not used to choose the final model.

> **Important caveat:** Removing identical feature vectors does not prove that all visually near-duplicate or correlated images were removed. The evaluation therefore measures performance under this dataset split, not necessarily across independent farms, acquisition sessions, or environments.

## 3. Methodology

### 3.1 Image preprocessing

Images were loaded with OpenCV and resized to **64 × 64 pixels** using the default `cv2.resize` interpolation (**INTER_LINEAR**). RGB, grayscale, and HSV representations were then computed.

**Implementation-critical detail:** A discarded experimental cell used `INTER_AREA`. The saved final model was trained using the **INTER_LINEAR** extractor. The deployed extractor was checked against **20 saved training examples**, and all reconstructed feature vectors matched the saved features to the reported precision.

### 3.2 Handcrafted feature extraction

| Feature family | Details | Number of features |
|:--|:--|--:|
| HSV histograms | 16 bins per H, S, and V channel | 48 |
| RGB statistics | Mean and standard deviation for each RGB channel | 6 |
| Uniform LBP | 8 neighbors, radius 1, 10-bin histogram | 10 |
| HOG | 8 orientations; 8×8 pixel cells; 2×2 cell blocks; L2-Hys | 1,568 |
| **Total** | | **1,632** |

### 3.3 Feature selection

An **ExtraTrees** model trained on the training subset supplied feature-importance scores. The **350 highest-ranked features** were selected using the saved index array `lgbm_selected_features.npy`. This reduced input dimensionality and improved the speed of subsequent LightGBM experiments.

In a validation ablation at **300 LightGBM estimators**, the 350-feature version achieved **94.6671% accuracy / 94.3177% macro F1**, compared with **94.2793% / 93.6643%** when using all 1,632 features. The recorded training times were approximately **106 seconds versus 530 seconds**, respectively, in the original Kaggle environment.

### 3.4 Models investigated

The experiment notebook explored SGD-based linear classification, ExtraTrees, a PCA/SVM pilot, LightGBM, and probability-averaging ensembles. Additional spatial and multiscale feature pilots were evaluated but not adopted in the final system.

**Selected classifier:** `LGBMClassifier`, with the following key settings:

```python
LGBMClassifier(
    objective="multiclass",
    num_class=15,
    n_estimators=850,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    colsample_bytree=0.85,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=3.0,
    n_jobs=2,
    random_state=42,
    verbosity=-1,
)
```

An **internal split of the training data** was used for early-stopping assessment; it did not stop before the maximum of 850 iterations. The selected model was then fitted on the full training subset and assessed on validation data.

## 4. Validation Results and Model Selection

The following scores were recorded in the **original Kaggle notebook** on the validation set. They are **not test scores**.

| Model | Validation accuracy | Validation macro F1 |
|:--|--:|--:|
| SGD baseline | 76.79% | 74.11% |
| ExtraTrees baseline | 90.47% | 89.72% |
| Tuned ExtraTrees (150 estimators) | 92.44% | 91.85% |
| LightGBM (earlier 350-feature model) | 94.93% | 94.52% |
| **LightGBM 850 — selected** | **95.22%** | **94.85%** |
| LightGBM 850 + ExtraTrees ensemble (50/50) | 95.18% | 94.91% |

The ensemble marginally improved **macro F1**, but the standalone LightGBM 850 was chosen for deployment because it achieved slightly higher validation accuracy with a simpler inference pipeline and fewer serialized model dependencies. These comparisons were conducted using validation data; the final test set remained reserved.

## 5. Independent Test Evaluation

The frozen model was evaluated on **3,094 held-out test images** after model selection.

| Metric | Final result |
|:--|--:|
| **Accuracy** | **95.1842%** |
| **Macro F1** | **93.8936%** |
| Macro precision | 94.4751% |
| Macro recall | 93.3954% |
| Correct predictions | 2,945 |
| Incorrect predictions | 149 |
| Test images | 3,094 |

The test accuracy was close to the validation accuracy, while macro F1 was somewhat lower. This difference underscores the importance of evaluating performance across classes, including smaller or more difficult categories. For example, **Tomato Early Blight** had test recall of **83.33%**, while **Potato Healthy** had **23 test examples** and a class F1 of approximately **81.82%**.

The saved evaluation artifacts include:

- `final_evaluation/test_metrics.json`
- `final_evaluation/classification_report.csv`
- `final_evaluation/confusion_matrix.csv`
- `final_evaluation/test_predictions.csv`

These files contain the recorded test evaluation; the clean notebook displays the frozen results without re-tuning or repeatedly testing the model.

## 6. Streamlit Application

The published application supports:

- Uploading a leaf image for classification.
- Displaying the predicted **plant** and **health condition**.
- Showing the **three highest-scoring classes** and model-estimated probabilities.
- Displaying an uncertainty notice for less decisive predictions.

**Repository:** https://github.com/nada-ali-13/PlantDoctor-AI

The live application is deployed on Streamlit Community Cloud. Add its exact URL here once confirmed: **[`[Live application URL]] (https://plantdoctor-ai-nada.streamlit.app/)**.

### Files needed for inference

```text
PlantDoctor-AI/
├── app.py
├── feature_extractor.py
├── lightgbm_850.joblib
├── lgbm_selected_features.npy
├── label_encoder.joblib
├── requirements.txt
├── README.md
└── PlantDoctor_Final.ipynb
```

The deployment does **not** need the original 20,638 training images or the complete cached feature matrix. The full feature matrix, original metadata, and split indices are required only for deeper research reproducibility and evaluation.

### Local execution

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

For reliable loading of serialized models, match the Python and library versions used in the Kaggle training environment where practical (see the saved `environment_info.json`). Only load `.joblib` artifacts from trusted sources.

### Reproducing the research notebook

1. Add the Kaggle PlantVillage dataset to the notebook environment.
2. Restore `all_features.npy` and `all_metadata.csv` from the separately saved **feature backup**.
3. Restore `split_indices.npz`, `label_encoder.joblib`, `lgbm_selected_features.npy`, `lightgbm_850.joblib`, and `final_evaluation/` from the **deployment backup**.
4. Open `PlantDoctor_Final.ipynb` and execute the cells in order.

**Note:** The cleaned submission notebook intentionally skips costly feature regeneration and model retraining by default. It documents the original training recipe and loads the frozen artifacts. The original experimental notebook should be retained separately as a historical record of all trials.

## 7. External Image Checks

The Streamlit app was manually tested with external example images. It correctly identified one **tomato early-blight** example with a model-estimated top-class probability of **97.4%**. In another external example labeled **potato late blight**, the system predicted **tomato late blight** with **80.3%** top-class probability: the disease name matched, but the **combined crop–disease class was incorrect**.

These are **illustrative individual cases**, not a statistically representative external benchmark. In particular, **97.4% and 80.3% are model-estimated class probabilities, not empirical accuracy rates**, and probability calibration has not been established.

## 8. Limitations and Future Work

1. **Domain shift:** PlantVillage-style images differ from field photographs in lighting, backgrounds, camera quality, occlusion, and leaf orientation. Real-world accuracy may be lower.
2. **Cross-crop confusion:** Similar diseases may affect multiple crops. The potato-versus-tomato late-blight example demonstrates this failure mode.
3. **Known-class restriction:** The model predicts one of 15 classes even if the image contains an unsupported plant, an unseen disease, or no leaf at all.
4. **Uncalibrated confidence:** High predicted probability does not guarantee correctness. Any warning threshold is a user-interface heuristic until independently validated.
5. **Dataset dependence:** Near-duplicate images or acquisition correlations may remain despite exact feature deduplication.
6. **Future improvement (not implemented):** Offer an optional **user-selected crop type**, while preserving automatic plant detection. Restricting predictions to a known crop may reduce cross-crop errors but requires independent validation.
7. **Additional validation:** Evaluate on a labeled, diverse set of real-world field images and investigate confidence calibration and out-of-distribution detection.

## 9. Conclusion

PlantDoctor AI demonstrates that a traditional machine-learning pipeline—combining **HSV, RGB, LBP, and HOG** features, **ExtraTrees-based feature selection**, and **LightGBM**—can classify 15 PlantVillage crop-and-condition classes with **95.18% independent test accuracy** and **93.89% macro F1**. The final extractor was validated against cached training features, and the classifier was integrated into a functioning Streamlit application. The system is a strong proof of concept, with further field testing required before real-world diagnostic use.

---

**Project:** PlantDoctor AI  
**Approach:** Traditional Machine Learning · Computer Vision · LightGBM · Streamlit  
**Code:** [github.com/nada-ali-13/PlantDoctor-AI](https://github.com/nada-ali-13/PlantDoctor-AI)
