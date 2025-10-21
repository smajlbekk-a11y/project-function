# src/compare_plots.py
import pandas as pd
import matplotlib.pyplot as plt

activations = ['sigmoid', 'tanh', 'relu']
metrics = ['train_loss', 'val_acc', 'grad_norm', 'epoch_time_sec']

for metric in metrics:
    plt.figure(figsize=(7, 4))
    for act in activations:
        data = pd.read_csv(f"results/{act}_history.csv")
        plt.plot(data['epoch'], data[metric], label=act.upper())
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Comparison — {metric}")
    plt.legend(); plt.grid()
    plt.savefig(f"results/compare_{metric}.png")

print("✅ Combined comparison plots saved in results/")
