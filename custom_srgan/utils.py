import os
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize
from PIL import Image
import torchvision.transforms as transforms

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved to {path}")

