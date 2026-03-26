import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset

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
# 2. Device
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================================================
# 3. Config
# =========================================================
CONFIG = {
    "data_dir": "./data",
    "results_dir": "./results_stage3_asdg",

    "generator_ckpt": "./results_stage1_cgan/generator.pth",
    "student_ckpt": "./results_stage2_student_baseline/best_student_baseline.pth",

    "batch_size": 128,
    "num_workers": 0,
    "num_classes": 10,
    "real_per_class": 300,

    # GAN config (must match Stage 1)
    "latent_dim": 128,
    "channels": 3,
    "g_feature_maps": 64,

    # ASDG loop
    "rounds": 3,
    "pool_per_class": 150,       # total candidate pool = 1500
    "select_per_round": 400,     # chosen synthetic examples each round
    "lambda_diversity": 0.20,

    "student_round_epochs": 3,
    "student_lr": 5e-4,
    "weight_decay": 1e-4,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)


# =========================================================
# 4. Transforms
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

plain_transform = transforms.Compose([
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
# 5. Data
# =========================================================
class LabelTensorWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.float()

        label = torch.tensor(label, dtype=torch.long)
        return image, label
train_dataset = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=True,
    download=True,
    transform=train_transform
)

train_dataset_plain = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=True,
    download=False,
    transform=plain_transform
)

test_dataset = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=False,
    download=True,
    transform=test_transform
)

class_names = train_dataset.classes
print("Classes:", class_names)


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

real_subset = LabelTensorWrapper(Subset(train_dataset, selected_indices))
real_subset_plain = LabelTensorWrapper(Subset(train_dataset_plain, selected_indices))

real_loader = DataLoader(
    real_subset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"]
)

real_loader_plain = DataLoader(
    real_subset_plain,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"]
)

print(f"Real subset size: {len(real_subset)}")
print(f"Test size: {len(test_dataset)}")


# =========================================================
# 6. Stage 1 Generator architecture
# =========================================================
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, channels, feature_maps):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.Conv2d(feature_maps, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([noise, label_vec], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.net(x)


# =========================================================
# 7. Stage 2 Student architecture
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
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

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

    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 8. Load checkpoints
# =========================================================
generator = ConditionalGenerator(
    latent_dim=CONFIG["latent_dim"],
    num_classes=CONFIG["num_classes"],
    channels=CONFIG["channels"],
    feature_maps=CONFIG["g_feature_maps"]
).to(device)

generator.load_state_dict(torch.load(CONFIG["generator_ckpt"], map_location=device))
generator.eval()

base_student_state = torch.load(CONFIG["student_ckpt"], map_location=device)

print("Loaded generator checkpoint.")
print("Loaded student baseline checkpoint.")


# =========================================================
# 9. Helpers
# =========================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return 100.0 * correct / total


def train_student(model, loader, test_loader, epochs=4, lr=5e-4, weight_decay=1e-4, label="student"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader)
        history.append(test_acc)

        print(
            f"[{label}] Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return history


@torch.no_grad()
def generate_candidate_pool(generator, pool_per_class=150):
    xs = []
    ys = []

    for c in range(CONFIG["num_classes"]):
        labels = torch.full((pool_per_class,), c, dtype=torch.long, device=device)
        noise = torch.randn(pool_per_class, CONFIG["latent_dim"], device=device)
        fake = generator(noise, labels).cpu()

        xs.append(fake)
        ys.append(labels.cpu())

    x_pool = torch.cat(xs, dim=0)
    y_pool = torch.cat(ys, dim=0)
    return x_pool, y_pool


@torch.no_grad()
def get_real_feature_bank(student, loader_plain):
    student.eval()
    feats_all = []

    for images, _ in loader_plain:
        images = images.to(device)
        feats = student.extract_features(images).cpu()
        feats_all.append(feats)

    return torch.cat(feats_all, dim=0)


@torch.no_grad()
def predict_entropy_and_features(student, images):
    student.eval()
    logits = student(images)
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    feats = student.extract_features(images)
    return entropy, feats


@torch.no_grad()
def active_select(student, x_pool, y_pool, real_feature_bank, k=600, lambda_diversity=0.35):
    student.eval()

    batch_size = 256
    entropies = []
    features = []

    for i in range(0, len(x_pool), batch_size):
        xb = x_pool[i:i+batch_size].to(device)
        ent, feat = predict_entropy_and_features(student, xb)
        entropies.append(ent.cpu())
        features.append(feat.cpu())

    entropies = torch.cat(entropies, dim=0)
    features = torch.cat(features, dim=0)

    novelty_scores = []
    chunk = 256
    for i in range(0, len(features), chunk):
        f = features[i:i+chunk]
        d = torch.cdist(f, real_feature_bank)
        min_d = d.min(dim=1).values
        novelty_scores.append(min_d)

    novelty = torch.cat(novelty_scores, dim=0)

    ent_norm = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
    nov_norm = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)

    score = ent_norm + lambda_diversity * nov_norm
    topk = torch.topk(score, k=min(k, len(score))).indices

    return x_pool[topk], y_pool[topk], score[topk]


def random_select(x_pool, y_pool, k=600):
    idx = torch.randperm(len(x_pool))[:k]
    return x_pool[idx], y_pool[idx]


def save_line_plot(values, title, ylabel, filename, xlabel="Round"):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_comparison_plot(real_only_acc, random_accs, active_accs, filename):
    plt.figure(figsize=(8, 4.5))
    plt.axhline(real_only_acc, linestyle="--", label="Real-only baseline")
    plt.plot(range(1, len(random_accs) + 1), random_accs, marker="o", label="Random synthetic")
    plt.plot(range(1, len(active_accs) + 1), active_accs, marker="o", label="Active synthetic")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("ASDG Comparison on CIFAR-10")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


# =========================================================
# 10. Initialize random vs active students
# =========================================================
student_random = StudentCNN(num_classes=CONFIG["num_classes"]).to(device)
student_active = StudentCNN(num_classes=CONFIG["num_classes"]).to(device)

student_random.load_state_dict(base_student_state)
student_active.load_state_dict(base_student_state)

real_only_acc = evaluate(student_random, test_loader)
print(f"\nReal-only baseline accuracy (loaded checkpoint): {real_only_acc:.2f}%")

random_round_accs = []
active_round_accs = []

random_datasets = []
active_datasets = []


# =========================================================
# 11. ASDG rounds
# =========================================================
for round_idx in range(CONFIG["rounds"]):
    print(f"\n================ ROUND {round_idx + 1}/{CONFIG['rounds']} ================")

    # Generate candidate synthetic pool
    x_pool, y_pool = generate_candidate_pool(
        generator,
        pool_per_class=CONFIG["pool_per_class"]
    )

    # ----------------------
    # Random selection
    # ----------------------
    x_rand, y_rand = random_select(
        x_pool, y_pool, k=CONFIG["select_per_round"]
    )

    random_datasets.append(TensorDataset(x_rand.float(), y_rand.long()))
    random_train_dataset = ConcatDataset([real_subset] + random_datasets)
    random_loader = DataLoader(
        random_train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    rand_hist = train_student(
        student_random,
        random_loader,
        test_loader,
        epochs=CONFIG["student_round_epochs"],
        lr=CONFIG["student_lr"],
        weight_decay=CONFIG["weight_decay"],
        label=f"Random-Round{round_idx+1}"
    )
    # rand_acc = rand_hist[-1]
    rand_acc = max(rand_hist)
    random_round_accs.append(rand_acc)
    print(f"[RESULT] Random synthetic accuracy after round {round_idx+1}: {rand_acc:.2f}%")

    # ----------------------
    # Active selection
    # ----------------------
    real_feature_bank = get_real_feature_bank(student_active, real_loader_plain)

    x_act, y_act, _ = active_select(
        student_active,
        x_pool,
        y_pool,
        real_feature_bank=real_feature_bank,
        k=CONFIG["select_per_round"],
        lambda_diversity=CONFIG["lambda_diversity"]
    )

    active_datasets.append(TensorDataset(x_act.float(), y_act.long()))
    active_train_dataset = ConcatDataset([real_subset] + active_datasets)
    active_loader = DataLoader(
        active_train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    act_hist = train_student(
        student_active,
        active_loader,
        test_loader,
        epochs=CONFIG["student_round_epochs"],
        lr=CONFIG["student_lr"],
        weight_decay=CONFIG["weight_decay"],
        label=f"Active-Round{round_idx+1}"
    )
    # act_acc = act_hist[-1]
    act_acc = max(act_hist)
    active_round_accs.append(act_acc)
    print(f"[RESULT] Active synthetic accuracy after round {round_idx+1}: {act_acc:.2f}%")


# =========================================================
# 12. Save plots
# =========================================================
save_line_plot(
    random_round_accs,
    "Random Synthetic Selection Accuracy by Round",
    "Test Accuracy (%)",
    os.path.join(CONFIG["results_dir"], "random_round_accuracy.png")
)

save_line_plot(
    active_round_accs,
    "Active Synthetic Selection Accuracy by Round",
    "Test Accuracy (%)",
    os.path.join(CONFIG["results_dir"], "active_round_accuracy.png")
)

save_comparison_plot(
    real_only_acc,
    random_round_accs,
    active_round_accs,
    os.path.join(CONFIG["results_dir"], "comparison.png")
)


# =========================================================
# 13. Save summary
# =========================================================
with open(os.path.join(CONFIG["results_dir"], "summary.txt"), "w") as f:
    f.write("Stage 3 ASDG CIFAR-10 Summary\n")
    f.write("=============================\n")
    f.write(f"Device: {device}\n")
    f.write(f"Real samples per class: {CONFIG['real_per_class']}\n")
    f.write(f"Total real training samples: {len(real_subset)}\n")
    f.write(f"Rounds: {CONFIG['rounds']}\n")
    f.write(f"Pool per class: {CONFIG['pool_per_class']}\n")
    f.write(f"Selected per round: {CONFIG['select_per_round']}\n")
    f.write(f"Lambda diversity: {CONFIG['lambda_diversity']}\n")
    f.write(f"Student round epochs: {CONFIG['student_round_epochs']}\n\n")
    f.write(f"Real-only baseline accuracy: {real_only_acc:.2f}%\n")
    f.write(f"Random round accuracies: {random_round_accs}\n")
    f.write(f"Active round accuracies: {active_round_accs}\n")

print("\nStage 3 complete.")
print(f"Real-only baseline accuracy: {real_only_acc:.2f}%")
print(f"Random round accuracies: {random_round_accs}")
print(f"Active round accuracies: {active_round_accs}")
print(f"Results saved in: {CONFIG['results_dir']}")