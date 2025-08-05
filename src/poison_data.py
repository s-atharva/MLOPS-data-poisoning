import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np


def poison_labels(y, noise_level):
    n_samples = len(y)
    n_noisy = int(n_samples * noise_level)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    y_noisy = y.copy()
    classes = np.unique(y)
    for idx in noisy_indices:
        old = y_noisy[idx]
        choices = classes[classes != old]
        y_noisy[idx] = np.random.choice(choices)
    return y_noisy


def generate_poisoned_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    for noise in [0.05, 0.10, 0.50]:
        y_poisoned = poison_labels(y, noise)
        data = pd.concat([X, y_poisoned], axis=1)
        data.to_csv(f"data/train_{int(noise * 100)}.csv", index=False)
        print(f"Saved poisoned dataset with {int(noise * 100)}% noise.")


if __name__ == "__main__":
    generate_poisoned_data()
