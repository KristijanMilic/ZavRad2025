import torch
from torchvision import transforms
from PIL import Image
import os
from custom_srgan.model import Generator


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((96, 96), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def save_image(tensor, filename):
    image = tensor.squeeze(0).detach().cpu()
    image = (image * 0.5) + 0.5  # Denormalize to [0, 1]
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(filename)


def upscale(image_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = Generator().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    lr_image = load_image(image_path).to(device)

    with torch.no_grad():
        sr_image = model(lr_image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(sr_image, output_path)
    print(f"Upscaled image saved to {output_path}")
