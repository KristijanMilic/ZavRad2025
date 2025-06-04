import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16, upscale_factor=2):
        super(Generator, self).__init__()
        self.upscale_factor = upscale_factor

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        self.conv_post_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        num_upsample_blocks = int(torch.log2(torch.tensor(upscale_factor)))
        self.upsample = nn.Sequential(
            *[UpsampleBlock(64, 2) for _ in range(num_upsample_blocks)]
        )

        self.output = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        residual = self.residuals(initial)
        post_residual = self.conv_post_res(residual)
        merged = initial + post_residual
        upsampled = self.upsample(merged)
        return torch.tanh(self.output(upsampled))

class Discriminator(nn.Module):
    def __init__(self, input_size=96):
        super(Discriminator, self).__init__()
        def block(in_channels, out_channels, stride):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *block(64, 64, 2),
            *block(64, 128, 1),
            *block(128, 128, 2),
            *block(128, 256, 1),
            *block(256, 256, 2),
            *block(256, 512, 1),
            *block(512, 512, 2),
        )

        fc_input_size = (input_size // 16) ** 2 * 512
        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.model(x)
        return self.adv_head(features)

