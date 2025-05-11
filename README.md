# PIRVision: Human Presence Detection Using PIR Sensor Arrays 🕵️‍♂️

> A supervised learning project for detecting human presence using PIR sensors, PCA, and Random Forest.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [Author](#author)

---

## 📌 Overview

**PIRVision** is a machine learning pipeline that classifies indoor human presence using readings from an array of 55 Passive Infrared (PIR) sensors and ambient temperature data. It identifies three activity classes:
- `0`: Vacancy
- `1`: Stationary Human Presence
- `3`: Other Activity/Motion

The project applies:
- Data preprocessing & cleaning
- Dimensionality reduction (PCA)
- Outlier detection (Isolation Forest)
- Supervised classification models (Random Forest, SVM, Logistic Regression)

---

## 📁 Dataset

- **Format**: Multivariate time series (4s intervals)
- **Features**: 
  - 55 PIR sensors: `PIR_1` to `PIR_55`
  - Ambient temperature (`Temperature_F`)
  - Timestamp: `Date`, `Time`
  - Label: Activity class (`0`, `1`, `3`)
- **Issues Addressed**:
  - Class imbalance
  - High dimensionality
  - Sensor noise

---

## 🛠️ Project Pipeline

1. **Preprocessing**:
   - Merge date and time
   - Clean missing values
   - Handle class imbalance using `class_weight='balanced'`
   - Standardize all features

2. **Dimensionality Reduction**:
   - PCA reduces 55 features to 10 principal components (retaining 95% variance)

3. **Outlier Detection**:
   - Isolation Forest applied to label `3` ("Other Activity") to remove noisy samples

4. **Classification Models**:
   - Logistic Regression
   - Support Vector Machine (RBF)
   - Random Forest (optimized with GridSearchCV)

5. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall, F1-score (per class)
   - Confusion matrices
   - Feature importance

---

## 💾 Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd pirvision

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## 🚀 Usage
Open and run the notebook:
```bash
jupyter notebook Project2_Bonus_Supervised.ipynb
```
Key notebook sections:
- Data Loading & Preprocessing
- PCA Dimensionality Reduction
- Outlier Detection with Isolation Forest
- Model Training & Evaluation

---
## 📊 Evaluation Results

The following table summarizes the classification performance of the three models used:

| Model                | Accuracy |
|---------------------|----------|
| Random Forest        | **97.4%** |
| SVM (RBF Kernel)     | ~96%     |
| Logistic Regression  | ~91%     |

- **Random Forest** achieved the best overall performance, with high F1-scores across all activity classes.
- **Isolation Forest** preprocessing notably improved classification of **Label 3 (Other Activity)** by removing approximately **10% noisy samples**, enhancing model robustness and reducing false positives.

---
## 📈 Visualizations
- PCA scatter plots
- Feature correlation heatmap
- Confusion matrices
- Feature importances (pre & post PCA)

---
## 🔮 Future Work
- Introduce deep learning models: CNNs, LSTMs, TCNs
- Self-supervised learning (SimCLR, MoCo)
- Domain adaptation (DANN, TCA)
- SMOTE for better minority class sampling
- Deploy in real-time smart environments

---
## 👩‍💻 Author
Mounika Seelam
Dept. of Computer Science
Kent State University


