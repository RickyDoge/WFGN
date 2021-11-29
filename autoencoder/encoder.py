import torch
from torch import nn
from torchvision import models as M


class AEEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(AEEncoder, self).__init__()
        pretrained_vgg = M.vgg11_bn(pretrained=pretrained)
        del pretrained_vgg.classifier
        del pretrained_vgg.avgpool

        self.encoder = nn.ModuleList()
        for module in pretrained_vgg.features:
            if isinstance(module, nn.MaxPool2d):
                self.encoder.append(nn.MaxPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    return_indices=True,
                ))
            else:
                self.encoder.append(module)

    def forward(self, x):
        pool_indices = []
        for module in self.encoder:
            y = module(x)
            if isinstance(module, nn.MaxPool2d):
                x = y[0]
                pool_indices.append(y[1])
            else:
                x = y

        return x, pool_indices
