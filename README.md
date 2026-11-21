# 🛡️ Insurance Risk Scoring Model

End-to-end deep learning system that predicts **insurance claim likelihood** from customer and vehicle data.
Final project for the Deep Learning unit.

---

## 📌 Problem
Given a customer's profile (age, driving experience, vehicle, history...), predict whether they will make an insurance claim (`OUTCOME`: 0 / 1).

**Dataset:** [Car Insurance Data (Kaggle)](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data). Download the CSV and place it in `data/raw/`.

---

## 👥 Team & Ownership

| # | Member | Part | Notebook |
|---|--------|------|----------|
| 1 | Abdulrhman | Data & Preprocessing | `notebooks/dev/01_data_preprocessing.ipynb` |
| 2 | Skkinner | Model Design & Regularization | `notebooks/dev/02_model_regularization.ipynb` |
| 3 | Sulaiman Allumaylim | Hyperparameter Tuning & Evaluation | `notebooks/dev/03_tuning_evaluation.ipynb` |
| 4 | Ahmed Alluqmani | Save/Load, Inference & Final Notebook | `notebooks/dev/04_save_load_inference.ipynb`, `src/` |

---

## 🗂️ Repo Structure

```
Deep_Learning_Project/
├── data/
│   ├── raw/              # original Kaggle CSV
│   └── processed/        # X_train.npy, X_val.npy, X_test.npy, y_*.npy  (from notebook 01)
├── models/               # preprocessor.pkl, best_model.keras
├── notebooks/
│   ├── dev/              # one notebook per member (01–04)
│   └── 00_final_pipeline.ipynb   # merged end-to-end notebook (final submission)
├── src/inference.py      # predict_risk() function
├── reports/figures/      # plots used in the presentation
├── requirements.txt
└── README.md
```

---

## 🔁 How the notebooks connect

```
01 Data ──► data/processed/*.npy + models/preprocessor.pkl
              │
              ├──► 02 Model ──► baseline & regularized models
              └──► 03 Tuning ──► models/best_model.keras + metrics
                                      │
                                      └──► 04 Inference ──► predict_risk()
```

---

## 📏 Team Rules

1. **Only edit your own notebook.** Notebooks don't merge well in git.
2. **Pull before you start, push when you finish** a working piece.
3. **Use the shared setup cell** (top of every notebook): same seed, same paths, same file names.
4. **The data split is frozen** once notebook 01 saves it. Don't re-split in other notebooks.
5. Every notebook must **run top to bottom** (Restart & Run All) before you push.
6. Save plots to `reports/figures/` so they can go straight into the slides.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Ahmed-log11/Deep_Learning_Project.git
cd Deep_Learning_Project
pip install -r requirements.txt
```

**Open in Colab:** File → Open notebook → GitHub tab → paste the repo URL.

---

## ✅ Required Components Checklist

- [x] ANN model
- [x] Regularization (L2, Dropout, Early Stopping)
- [x] Hyperparameter tuning (Keras Tuner)
- [x] Evaluation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- [x] Model saving & loading
- [x] Inference pipeline (`predict_risk()` function)
- [x] Final merged notebook (`notebooks/00_final_pipeline.ipynb`)

---

## 📊 Results (test set, 1,223 customers, threshold 0.5)

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Baseline ANN | 0.644 | 0.740 | 0.689 | 0.862 |
| Regularized ANN | 0.693 | 0.819 | 0.751 | 0.907 |
| **Tuned ANN (final model)** | 0.669 | 0.835 | 0.743 | 0.903 |

- **Final model:** the tuned ANN from notebook 03 (`models/best_model.keras`), chosen by the best **validation** AUC (0.939).
- Regularization gave the main improvement. Tuning matched the regularized model on test but did not beat it (difference within noise).

## ⚠️ Limitations
- 18.5% of rows were dropped because of missing values (claim rate unchanged, but less training data).
- Fixed 0.5 threshold: high recall (83%) but more false alarms (157 of 842 non-claimers flagged).
- Small dataset, one split, one seed: no cross-validation.
- `RACE` and `GENDER` are used as features, which raises fairness / legal concerns for real insurance pricing.
- `PAST_ACCIDENTS`, `SPEEDING_VIOLATIONS`, `DUIS` must be history *before* the policy, otherwise they leak the outcome; the dataset doesn't document this.
