import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# Pixel 归一化
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# Instance 归一化
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style_linear = nn.Linear(style_dim, in_channel * 2)  # 512 -> 1024

        self.style_linear.bias.data[:in_channel] = 1  # first half bias to 1
        self.style_linear.bias.data[in_channel:] = 0  # second half bias to 0

    def forward(self, input, style):
        style = self.style_linear(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)  # clear the current "style"
        out = gamma * out + beta  # transfer to target "style"
        return out


# 转置卷积 + 双线性插值
class FusedUpsample(nn.Module):
    # conv + bilinear upsample implement
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)
        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4
        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out


class NoiseInjection(nn.Module):
    def __init__(self, out_channel):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class Blur(nn.Module):
    def __init__(self, out_channel):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 20
        self.kernel = self.kernel.repeat(out_channel, 1, 1, 1)
        self.out_channel = out_channel

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.out_channel)


class GeneratorConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, num_dim=512, upsample=False, fused=False):
        super(GeneratorConvBlock, self).__init__()
        if upsample:
            if fused:
                self.conv1 = nn.Sequential(FusedUpsample(in_channel, out_channel, kernel_size, padding),
                                           Blur(out_channel))
            else:
                self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),  # 邻近插值
                                           Blur(out_channel))
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
            self.conv1.weight.data.normal_()

        self.noise1 = NoiseInjection(out_channel)
        self.relu1 = nn.LeakyReLU(0.2)
        self.adain1 = AdaptiveInstanceNorm(out_channel, num_dim)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)
        self.conv2.weight.data.normal_()
        self.noise2 = NoiseInjection(out_channel)
        self.relu2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaptiveInstanceNorm(out_channel, num_dim)

    def forward(self, x, style, noise):
        out = self.conv1(x)
        out = self.noise1(out, noise)
        out = self.relu1(out)
        out = self.adain1(out, style)
        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.relu2(out)
        out = self.adain2(out, style)
        return out


class StyleGenerator(nn.Module):
    def __init__(self, dim=512, input_channel=512, mlp_layers=8):
        super(StyleGenerator, self).__init__()
        n_mlp = [PixelNorm()]
        for _ in range(mlp_layers):
            n_mlp.append(nn.Linear(dim, dim))
            n_mlp[len(n_mlp) - 1].weight.data.normal_()
            n_mlp.append(nn.LeakyReLU(0.2))

        self.mapping_net = nn.Sequential(*n_mlp)
        self.constant_input = nn.Parameter(torch.randn(input_channel, 4, 4))
        self.upConv = nn.ModuleList([
            GeneratorConvBlock(512, 512, 3, 1),  # 4
            GeneratorConvBlock(512, 512, 3, 1, upsample=True),  # 8
            GeneratorConvBlock(512, 512, 3, 1, upsample=True),  # 16
            GeneratorConvBlock(512, 512, 3, 1, upsample=True),  # 32
            GeneratorConvBlock(512, 256, 3, 1, upsample=True, fused=True),  # 64
            GeneratorConvBlock(256, 128, 3, 1, upsample=True, fused=True),  # 128
            GeneratorConvBlock(128, 64, 3, 1, upsample=True, fused=True),  # 256
        ])
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.final.weight.data.normal_()

    def get_mean_style(self, latent_input):
        return self.mapping_net(latent_input).mean(0, keepdim=True)

    def forward(self, latent_input, style_weight=0., mean_style=None, noise=None):
        style = self.mapping_net(latent_input)  # batch * d -> batch * d
        if noise is None:
            noise = []
            for i in range(7):  # batch * 1 * 4 * 4
                width = 4 * (2 ** i)
                noise.append(torch.randn(latent_input.shape[0], 1, width, width, device=latent_input.device))
        if mean_style is not None:
            mean_style = mean_style.repeat(latent_input.shape[0], 1)  # batch * d
            style = mean_style + (style - mean_style) * style_weight

        x = self.constant_input.repeat(latent_input.shape[0], 1, 1, 1)  # batch * 512 * 4 * 4

        y = None
        for i, conv in enumerate(self.upConv):
            y = conv(x, style, noise[i]) if y is None else conv(y, style, noise[i])
        return self.final(y)
