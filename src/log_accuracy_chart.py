import mlflow
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('http://localhost:5000')

noise_levels = [0, 5, 10, 50]
accuracies = [1.00, 0.97, 0.89, 0.36]

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, accuracies, marker='o', linestyle='--')
plt.title("Accuracy vs. Label Noise in Iris Dataset")
plt.xlabel("Label Noise Level (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.ylim(0, 1.05)

plot_file = "accuracy_vs_noise.png"
plt.savefig(plot_file)
plt.close()

mlflow.set_experiment("IRIS Poison Detection")
with mlflow.start_run(run_name="accuracy_chart"):
    mlflow.log_artifact(plot_file)
    mlflow.log_param("chart", "Accuracy vs Label Noise")
    print("Chart logged to MLflow!")
