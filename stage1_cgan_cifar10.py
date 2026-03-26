import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

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
# 2. Device selection (Mac MPS / CUDA / CPU)
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
    "results_dir": "./results_stage1_cgan",
    "batch_size": 128,
    "num_workers": 0,         # safer on Mac
    "image_size": 32,
    "channels": 3,
    "num_classes": 10,
    "latent_dim": 128,
    "g_feature_maps": 64,
    "d_feature_maps": 64,
    "epochs": 60,
    "lr": 2e-4,
    "beta1": 0.5,
    "sample_every": 5,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)


# =========================================================
# 4. CIFAR-10 Data
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root=CONFIG["data_dir"],
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=False
)

class_names = train_dataset.classes
print("Classes:", class_names)


# =========================================================
# 5. Helper: initialize weights
# =========================================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# =========================================================
# 6. Conditional Generator
# =========================================================
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, channels, feature_maps):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            # input: [B, latent_dim + num_classes, 1, 1]
            nn.ConvTranspose2d(latent_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 4x4

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 32x32

            nn.Conv2d(feature_maps, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_vec = self.label_emb(labels)              # [B, num_classes]
        x = torch.cat([noise, label_vec], dim=1)        # [B, latent_dim + num_classes]
        x = x.unsqueeze(2).unsqueeze(3)                 # [B, C, 1, 1]
        return self.net(x)

# =========================================================
# 7. Conditional Discriminator
# =========================================================
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes, channels, feature_maps, image_size=32):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Label embedding maps label -> image plane
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        # Input channels = image channels + 1 label channel
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_map = self.label_emb(labels).view(labels.size(0), 1, self.image_size, self.image_size)
        x = torch.cat([images, label_map], dim=1)
        out = self.net(x)
        return out.view(-1, 1)


# =========================================================
# 8. Build models
# =========================================================
netG = ConditionalGenerator(
    latent_dim=CONFIG["latent_dim"],
    num_classes=CONFIG["num_classes"],
    channels=CONFIG["channels"],
    feature_maps=CONFIG["g_feature_maps"]
).to(device)

netD = ConditionalDiscriminator(
    num_classes=CONFIG["num_classes"],
    channels=CONFIG["channels"],
    feature_maps=CONFIG["d_feature_maps"],
    image_size=CONFIG["image_size"]
).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

print(netG)
print(netD)


# =========================================================
# 9. Loss and optimizers
# =========================================================
criterion = nn.BCELoss()

optimizerD = optim.Adam(
    netD.parameters(),
    lr=CONFIG["lr"],
    betas=(CONFIG["beta1"], 0.999)
)

optimizerG = optim.Adam(
    netG.parameters(),
    lr=CONFIG["lr"],
    betas=(CONFIG["beta1"], 0.999)
)


# =========================================================
# 10. Fixed noise for monitoring
# =========================================================
fixed_noise = torch.randn(100, CONFIG["latent_dim"], device=device)

fixed_labels = torch.tensor(
    [i for i in range(10) for _ in range(10)],
    dtype=torch.long,
    device=device
)


# =========================================================
# 11. Utility functions
# =========================================================
def denorm(x):
    return x * 0.5 + 0.5


def save_generated_grid(generator, epoch, save_path):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()
    grid = make_grid(denorm(fake), nrow=10, normalize=False)
    save_image(grid, save_path)
    generator.train()


def save_classwise_samples(generator, save_path, num_per_class=8):
    generator.eval()
    all_images = []
    with torch.no_grad():
        for c in range(CONFIG["num_classes"]):
            labels = torch.full((num_per_class,), c, dtype=torch.long, device=device)
            noise = torch.randn(num_per_class, CONFIG["latent_dim"], device=device)
            fake = generator(noise, labels).detach().cpu()
            all_images.append(fake)
    all_images = torch.cat(all_images, dim=0)
    grid = make_grid(denorm(all_images), nrow=num_per_class, normalize=False)
    save_image(grid, save_path)
    generator.train()


# =========================================================
# 12. Training loop
# =========================================================
g_losses = []
d_losses = []

print("\nStarting Training...\n")

for epoch in range(CONFIG["epochs"]):
    running_g_loss = 0.0
    running_d_loss = 0.0

    for i, (real_images, real_labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # -------------------------------------------------
        # Train Discriminator
        # -------------------------------------------------
        netD.zero_grad()

        # Real images
        output_real = netD(real_images, real_labels)
        d_loss_real = criterion(output_real, real_targets)

        # Fake images
        noise = torch.randn(batch_size, CONFIG["latent_dim"], device=device)
        fake_labels = torch.randint(0, CONFIG["num_classes"], (batch_size,), device=device)
        fake_images = netG(noise, fake_labels)

        output_fake = netD(fake_images.detach(), fake_labels)
        d_loss_fake = criterion(output_fake, fake_targets)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizerD.step()

        # -------------------------------------------------
        # Train Generator
        # -------------------------------------------------
        netG.zero_grad()

        output_fake_for_g = netD(fake_images, fake_labels)
        g_loss = criterion(output_fake_for_g, real_targets)

        g_loss.backward()
        optimizerG.step()

        running_d_loss += d_loss.item()
        running_g_loss += g_loss.item()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                f"Batch [{i}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    epoch_d_loss = running_d_loss / len(train_loader)
    epoch_g_loss = running_g_loss / len(train_loader)

    d_losses.append(epoch_d_loss)
    g_losses.append(epoch_g_loss)

    print(
        f"\nEpoch [{epoch+1}/{CONFIG['epochs']}] Completed | "
        f"Avg D Loss: {epoch_d_loss:.4f} | Avg G Loss: {epoch_g_loss:.4f}\n"
    )

    if (epoch + 1) % CONFIG["sample_every"] == 0 or epoch == 0:
        save_generated_grid(
            netG,
            epoch + 1,
            os.path.join(CONFIG["results_dir"], f"generated_epoch_{epoch+1}.png")
        )


# =========================================================
# 13. Save final sample grids
# =========================================================
save_generated_grid(
    netG,
    CONFIG["epochs"],
    os.path.join(CONFIG["results_dir"], "final_generated_grid.png")
)

save_classwise_samples(
    netG,
    os.path.join(CONFIG["results_dir"], "final_classwise_samples.png"),
    num_per_class=10
)


# =========================================================
# 14. Save loss plot
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Conditional GAN Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["results_dir"], "gan_loss_curve.png"), dpi=200)
plt.close()


# =========================================================
# 15. Save model checkpoints
# =========================================================
torch.save(netG.state_dict(), os.path.join(CONFIG["results_dir"], "generator.pth"))
torch.save(netD.state_dict(), os.path.join(CONFIG["results_dir"], "discriminator.pth"))


# =========================================================
# 16. Save training summary
# =========================================================
with open(os.path.join(CONFIG["results_dir"], "summary.txt"), "w") as f:
    f.write("Conditional GAN CIFAR-10 Training Summary\n")
    f.write("========================================\n")
    f.write(f"Device: {device}\n")
    f.write(f"Epochs: {CONFIG['epochs']}\n")
    f.write(f"Batch size: {CONFIG['batch_size']}\n")
    f.write(f"Latent dim: {CONFIG['latent_dim']}\n")
    f.write(f"Learning rate: {CONFIG['lr']}\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Final Generator Loss: {g_losses[-1]:.4f}\n")
    f.write(f"Final Discriminator Loss: {d_losses[-1]:.4f}\n")

print("\nTraining complete.")
print(f"Results saved in: {CONFIG['results_dir']}")