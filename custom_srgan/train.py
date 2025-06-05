import os
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from .model import Generator, Discriminator
from .loss import VGGLoss
from .utils import save_checkpoint
from .consts import *


class SRDataset(Dataset):
    def __init__(self, root_dir, upscale_factor=2):
        self.upscale_factor = upscale_factor
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        self.to_pil = transforms.ToPILImage()
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Ensure divisible by upscale_factor
        width, height = img.size
        new_width = width - (width % self.upscale_factor)
        new_height = height - (height % self.upscale_factor)
        img = img.resize((new_width, new_height), Image.BICUBIC)

        # HR is full-size image
        hr = self.hr_transform(img)

        # LR is downsampled
        lr_img = img.resize(
            (new_width // self.upscale_factor, new_height // self.upscale_factor),
            Image.BICUBIC
        )
        lr = self.lr_transform(lr_img)

        return lr, hr

def pad_to_largest(batch):
    lr_imgs, hr_imgs = zip(*batch)

    # Find max height and width from HR images
    max_h = max(img.shape[1] for img in hr_imgs)
    max_w = max(img.shape[2] for img in hr_imgs)

    lr_padded = []
    hr_padded = []

    for lr, hr in zip(lr_imgs, hr_imgs):
        _, h_hr, w_hr = hr.shape
        _, h_lr, w_lr = lr.shape

        # Calculate padding amounts (left, right, top, bottom)
        pad_hr = [
            (max_w - w_hr) // 2, (max_w - w_hr + 1) // 2,
            (max_h - h_hr) // 2, (max_h - h_hr + 1) // 2,
        ]
        pad_lr = [
            (max_w // 2 - w_lr) // 2, (max_w // 2 - w_lr + 1) // 2,
            (max_h // 2 - h_lr) // 2, (max_h // 2 - h_lr + 1) // 2,
        ]

        hr_pad = F.pad(hr, pad_hr, mode="reflect")
        lr_pad = F.pad(lr, pad_lr, mode="reflect")

        hr_padded.append(hr_pad)
        lr_padded.append(lr_pad)

    return torch.stack(lr_padded), torch.stack(hr_padded)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upscale_factor = 2  # or 4, depending on model config

    dataset = SRDataset("data/DIV2K_train_HR", upscale_factor=upscale_factor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_to_largest
    )

    generator = Generator(upscale_factor=upscale_factor).to(device)
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
                      f"G Loss: {g_loss.item():.4f} D Loss: {d_loss.item():.6f} "
                      f"LR shape: {lr.shape}, SR shape: {sr.shape}")

        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(generator, optimizer_G, epoch, f"checkpoints/generator.pth")
        save_checkpoint(discriminator, optimizer_D, epoch, f"checkpoints/discriminator.pth")

