import pandas as pd
import mlflow
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri('http://localhost:5000')


def train_model(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.set_experiment("IRIS Poison Detection")
    with mlflow.start_run(run_name=file_path):
        mlflow.log_param("data", file_path)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metrics({f"f1_class_{k}": v["f1-score"] for k, v in report.items() if k.isdigit()})
        mlflow.sklearn.log_model(clf, "model")
        print(f"Logged results for {file_path} to MLflow.")


if __name__ == "__main__":
    train_model("iris.csv")
