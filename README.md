Name  :- Neer Patel 


Roll no :- DA25M021

# GMM-Based Synthetic Sampling for Imbalanced Data

## 📌 Overview
This project implements **Gaussian Mixture Model (GMM)-based synthetic sampling** to address class imbalance in a credit card fraud detection dataset.  
A baseline Logistic Regression model is first trained on the **original imbalanced data**.  
Next, GMM is used to generate synthetic minority samples and combined with Clustering-Based Undersampling (CBU) of the majority class to create balanced datasets.  
The goal is to evaluate whether GMM-based oversampling improves fraud detection compared to the baseline.

---

## 1️⃣ Problem Statement
Financial fraud detection datasets are highly imbalanced: fraudulent transactions represent only a tiny fraction of the data.  
Training a classifier directly on such data often leads to high accuracy but **poor recall on the minority (fraud) class**.  
We aim to create a balanced training set using **GMM-based synthetic sampling** so the model can learn minority class patterns without overfitting.

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 2️⃣ Baseline Model
- **Model:** Logistic Regression  
- **Data split:** Stratified train/test, preserving the original imbalance.  
- **Reasoning:** Precision, Recall, and F1-score are more informative than accuracy because we care about correctly identifying fraudulent transactions.

---

## 3️⃣ GMM-Based Synthetic Sampling

### Theoretical Foundation
- **SMOTE vs. GMM:**  
  - *SMOTE* interpolates linearly between minority samples.  
  - *GMM* assumes the minority class follows a mixture of Gaussians and **samples from the learned probability distribution**, capturing complex, multi-modal shapes.
- **Why GMM?**  
  - Handles multiple sub-groups and non-linear boundaries better than simple interpolation.
  - Generates realistic synthetic points that follow the true underlying density.

### Implementation
1. **Fit GMM:**  
   - Only on minority-class training data.
   - Chose number of components (*k*) using **AIC** and **BIC** to balance model fit and complexity.
2. **Generate Synthetic Samples:**  
   - Sample from the fitted GMM until the minority count matches the majority.
3. **Combine Data:**  
   - Merge synthetic minority samples with original training data.
4. **Clustering-Based Undersampling (CBU):**  
   - Cluster the majority class and sample representative points to reduce its size.
   - Create a second balanced dataset with both GMM oversampling and CBU undersampling.

---

## 4️⃣ Model Training & Evaluation
Two new Logistic Regression models were trained:
- **GMM Balanced v1:** Original majority + GMM-sampled minority.
- **GMM + EBU v2:** Majority after CBU + GMM-sampled minority.

Evaluated on the **original, imbalanced test set** to mimic real-world deployment.

### 📊 Metrics

| Model            | Accuracy | Precision | Recall | F1-score |
|------------------|---------:|---------:|------:|--------:|
| Baseline LR      | **0.9989** | **0.7264** | 0.6260 | **0.6725** |
| GMM Balanced v1  | 0.9832 | 0.0837 | **0.8780** | 0.1529 |
| GMM + EBU v2     | 0.9722 | 0.0526 | **0.8862** | 0.0993 |

---

## 5️⃣ Analysis
- **Recall Surge:**  
  GMM-based sampling dramatically improved recall (~0.88 vs. 0.63), meaning far more fraud cases are caught.
- **Precision Drop:**  
  Precision fell sharply, indicating more false positives.
- **F1-score Tradeoff:**  
  Overall F1 decreased because of the precision–recall imbalance.
- **Accuracy Misleading:**  
  Accuracy declined slightly but remains high because the majority class dominates.

---

## 6️⃣ Final Recommendation
- **Use GMM when Recall is Critical:**  
  Fraud detection prioritizes catching fraud over minimizing false alarms. GMM sampling is effective in such high-stakes scenarios.
- **Not Ideal for Balanced F1:**  
  If false positives are costly, consider hybrid methods (e.g., ensemble models, cost-sensitive learning) to recover precision.
- **Data Characteristics Matter:**  
  GMM is best when the minority distribution truly contains multiple Gaussian-like subgroups. If the data is not multi-modal, simpler methods like SMOTE may suffice.

> ✅ **Conclusion:**  
> GMM-based synthetic sampling **significantly improves the ability to detect fraud** by boosting recall, making it preferable when detecting every fraudulent transaction is more important than achieving a high overall F1-score.

