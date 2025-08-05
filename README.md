# MLOPS-data-poisoning
## 🚀 What This Demo Does

- Creates 3 poisoned datasets with 5%, 10%, and 50% label corruption.
- Trains a `RandomForestClassifier` model on each poisoned dataset.
- Logs results (accuracy, classification report) to **MLflow**.
- Automates the training pipeline using **DVC** and **GitHub Actions**.

---

## 🧠 Why This Matters

This experiment shows how machine learning models can be sensitive to **incorrect or malicious labels**, and how performance degrades as label poisoning increases.

| Poison Level | Expected Behavior       |
|--------------|-------------------------|
| 5%           | Slight performance drop |
| 10%          | Noticeable impact       |
| 50%          | Major confusion         |

---