import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random, argparse, time, os, csv
import matplotlib.pyplot as plt
from model import MLP

# 🔁 reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_grad_norm(model):
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
            count += 1
    return total_norm / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--subset_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    # ✅ Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    subset, _ = random_split(dataset, [args.subset_size, len(dataset) - args.subset_size])
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_ds, val_ds = random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    # ✅ Model
    model = MLP(args.activation)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs('results', exist_ok=True)
    csv_path = f"results/{args.activation}_lr{args.lr}_history.csv"
    log_path = f"results/log_{args.activation}_lr{args.lr}.txt"

    history = {'epoch': [], 'train_loss': [], 'val_acc': [], 'grad_norm': [], 'epoch_time_sec': []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 Training started with activation = {args.activation.upper()}, learning_rate = {args.lr}\n")

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_grad_norm = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            grad_norm = compute_grad_norm(model)
            total_grad_norm += grad_norm
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        elapsed = time.time() - start_time

        # ✅ Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        # ✅ Save metrics
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['grad_norm'].append(avg_grad_norm)
        history['epoch_time_sec'].append(elapsed)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"GradNorm: {avg_grad_norm:.4f} | Time: {elapsed:.2f}s")

        # Log to text
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{val_acc:.4f},{avg_grad_norm:.4f},{elapsed:.2f}\n")

        # ✅ Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"⏹️ Early stopping at epoch {epoch+1} — no improvement for {args.early_stop_patience} epochs.")
            break

        # ✅ Stop if gradients explode
        if avg_grad_norm > 10.0:
            print(f"💥 Stopped early — gradients exploded (‖∇‖={avg_grad_norm:.2f})")
            break

    # ✅ Save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))

    # ✅ Plot metrics
    for metric, ylabel in zip(
        ['train_loss', 'val_acc', 'grad_norm', 'epoch_time_sec'],
        ['Loss', 'Accuracy', '‖∇‖', 'Seconds']
    ):
        plt.figure(figsize=(6, 4))
        plt.plot(history['epoch'], history[metric], label=f"{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{args.activation.upper()} (lr={args.lr}) — {metric}")
        plt.legend(); plt.grid(); plt.tight_layout()
        plt.savefig(f"results/{args.activation}_lr{args.lr}_{metric}.png")

    print(f"\n✅ Training complete! Results saved to {csv_path} and plots to 'results/'.\n")


if __name__ == "__main__":
    main()
