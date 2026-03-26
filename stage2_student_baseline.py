import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


set_seed(42)


# =========================================================
# 2. Device selection
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================================================
# 3. Configuration
# =========================================================
CONFIG = {
    "data_dir": "./data",
    "results_dir": "./results_stage2_student_baseline",
    "batch_size": 128,
    "num_workers": 0,   # safer on Mac
    "num_classes": 10,
    "real_per_class": 300,
    "epochs": 25,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)


# =========================================================
# 4. Data transforms
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# =========================================================
# 5. Load CIFAR-10
# =========================================================
train_dataset = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=False,
    download=True,
    transform=test_transform
)

class_names = train_dataset.classes
print("Classes:", class_names)


# =========================================================
# 6. Build limited real-data subset
# =========================================================
def build_few_shot_subset(dataset, per_class=300):
    targets = np.array(dataset.targets)
    selected_indices = []

    for c in range(10):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        selected_indices.extend(idx[:per_class].tolist())

    np.random.shuffle(selected_indices)
    return selected_indices


selected_indices = build_few_shot_subset(
    train_dataset,
    per_class=CONFIG["real_per_class"]
)

real_subset = Subset(train_dataset, selected_indices)

train_loader = DataLoader(
    real_subset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=False
)

print(f"Real subset size: {len(real_subset)}")
print(f"Test size: {len(test_dataset)}")


# =========================================================
# 7. Student CNN
# =========================================================
class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = StudentCNN(num_classes=CONFIG["num_classes"]).to(device)
print(model)


# =========================================================
# 8. Loss / optimizer
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]
)


# =========================================================
# 9. Training / evaluation helpers
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# =========================================================
# 10. Main training loop
# =========================================================
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_test_acc = 0.0

print("\nStarting Student Baseline Training...\n")

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{CONFIG['epochs']}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(
            model.state_dict(),
            os.path.join(CONFIG["results_dir"], "best_student_baseline.pth")
        )


# =========================================================
# 11. Save final model
# =========================================================
torch.save(
    model.state_dict(),
    os.path.join(CONFIG["results_dir"], "final_student_baseline.pth")
)


# =========================================================
# 12. Save plots
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Student Baseline Loss Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["results_dir"], "loss_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Student Baseline Accuracy Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["results_dir"], "accuracy_curve.png"), dpi=200)
plt.close()


# =========================================================
# 13. Save summary
# =========================================================
with open(os.path.join(CONFIG["results_dir"], "summary.txt"), "w") as f:
    f.write("Student Baseline CIFAR-10 Summary\n")
    f.write("=================================\n")
    f.write(f"Device: {device}\n")
    f.write(f"Real samples per class: {CONFIG['real_per_class']}\n")
    f.write(f"Total real training samples: {len(real_subset)}\n")
    f.write(f"Epochs: {CONFIG['epochs']}\n")
    f.write(f"Batch size: {CONFIG['batch_size']}\n")
    f.write(f"Learning rate: {CONFIG['lr']}\n")
    f.write(f"Weight decay: {CONFIG['weight_decay']}\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Best Test Accuracy: {best_test_acc:.2f}%\n")
    f.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")


print("\nTraining complete.")
print(f"Best Test Accuracy: {best_test_acc:.2f}%")
print(f"Results saved in: {CONFIG['results_dir']}")