# Tumor classifier: logistic regression vs random forest

A small, reproducible machine learning project. It trains two classifiers on the **Breast Cancer Wisconsin (Diagnostic)** dataset that ships with scikit-learn (569 samples, 30 numeric features) and compares them with cross-validation and a held-out test set.

> This is a learning and portfolio project. It is **not a medical device** and must not be used for diagnosis.

## Results

Fixed random seed (42), stratified 75/25 split (426 train, 143 test), 5-fold cross-validation on the training set only.

| Model | CV ROC-AUC (mean ± std) | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (scaled) | 0.996 ± 0.004 | 0.986 | 0.989 | 0.998 |
| Random Forest (300 trees) | 0.988 ± 0.011 | 0.958 | 0.978 | 0.995 |

Plots are written to `outputs/`: `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`. Raw numbers are in `outputs/metrics.json`.

## Run it

```bash
pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -t . -v
```

## What this does and does not show

- The simple, scaled linear model beat the more complex forest here. On small, clean tabular data this is common, and it is a reminder to start with a baseline.
- The test set has only 143 rows, so one or two extra mistakes move accuracy by about 1 percentage point. Treat differences between the models as small.
- The dataset is clean and well studied. Real client data usually needs cleaning, missing-value handling and leakage checks first.
- No hyperparameter search was done, and no external validation set was used.

## Next steps

- Add a hyperparameter search and calibrate the probabilities.
- Try the same pipeline on a messier real-world dataset (for example customer churn).
- Wrap the model in a small web app (Django) with input validation.

MIT licensed.
