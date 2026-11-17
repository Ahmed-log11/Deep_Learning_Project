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
| 1 | _name_ | Data & Preprocessing | `notebooks/dev/01_data_preprocessing.ipynb` |
| 2 | _name_ | Model Design & Regularization | `notebooks/dev/02_model_regularization.ipynb` |
| 3 | _name_ | Hyperparameter Tuning & Evaluation | `notebooks/dev/03_tuning_evaluation.ipynb` |
| 4 | _name_ | Save/Load, Inference & Repo | `notebooks/dev/04_save_load_inference.ipynb`, `src/` |

---

## 🗂️ Repo Structure

```
insurance-risk-scoring/
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
git clone <repo-url>
cd insurance-risk-scoring
pip install -r requirements.txt
```

**Open in Colab:** File → Open notebook → GitHub tab → paste the repo URL.

---

## ✅ Required Components Checklist

- [ ] ANN model
- [ ] Regularization (L2, Dropout, Early Stopping)
- [ ] Hyperparameter tuning (Keras Tuner)
- [ ] Evaluation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- [ ] Model saving & loading
- [ ] Inference pipeline (`predict_risk()` function)
- [ ] Final merged notebook
- [ ] Presentation

---

## 📊 Results
_To be filled after evaluation._

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Baseline ANN | | | | |
| Regularized ANN | | | | |
| Tuned ANN | | | | |

## ⚠️ Limitations
_To be filled._
