import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=35):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_index])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)

class VGGLoss(nn.Module):
    def __init__(self, feature_layer=22):
        super(VGGLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(layer_index=feature_layer)
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        sr = (sr + 1.0) / 2.0
        hr = (hr + 1.0) / 2.0
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.criterion(sr_features, hr_features)