import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from .model import Generator, Discriminator
from .loss import VGGLoss
from .utils import save_checkpoint
from .consts import *


class SRDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.hr_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((48, 48), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)
        return lr, hr

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SRDataset("data/DIV2K_train_HR")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    content_loss_fn = VGGLoss().to(device)
    adversarial_loss_fn = nn.BCELoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=GENERATOR_LR, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.9, 0.999))

    for epoch in range(NUM_EPOCHS):
        for i, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            valid = torch.ones((lr.size(0), 1), device=device)
            fake = torch.zeros((lr.size(0), 1), device=device)

            # Train Generator
            optimizer_G.zero_grad()
            sr = generator(lr)
            pred_fake = discriminator(sr)
            content_loss = content_loss_fn(sr, hr)
            adv_loss = adversarial_loss_fn(pred_fake, valid)
            g_loss = content_loss + 1e-3 * adv_loss
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(hr)
            pred_fake = discriminator(sr.detach())
            real_loss = adversarial_loss_fn(pred_real, valid)
            fake_loss = adversarial_loss_fn(pred_fake, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                      f"G Loss: {g_loss.item():.4f} D Loss: {d_loss.item():.6f}")

        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(generator, optimizer_G, epoch, f"checkpoints/generator.pth")
        save_checkpoint(discriminator, optimizer_D, epoch, f"checkpoints/discriminator.pth")