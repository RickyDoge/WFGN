import torch
from torch import nn


class AEDecoder(nn.Module):
    def __init__(self, encoder):
        super(AEDecoder, self).__init__()
        self.decoder = nn.ModuleList()
        for module in reversed(encoder.encoder):
            if isinstance(module, nn.Conv2d):
                self.decoder.append(nn.ConvTranspose2d(
                    in_channels=module.out_channels,
                    out_channels=module.in_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                ))
                self.decoder.append(nn.BatchNorm2d(module.in_channels))
                self.decoder.append(nn.ReLU(inplace=True))
            elif isinstance(module, nn.MaxPool2d):
                self.decoder.append(nn.MaxUnpool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                ))

    def forward(self, x, pool_indices):
        pool_indices, ptr = list(reversed(pool_indices)), 0
        for module in self.decoder:
            if isinstance(module, nn.MaxUnpool2d):
                x = module(x, indices=pool_indices[ptr])
                ptr += 1
            else:
                x = module(x)
        return x
