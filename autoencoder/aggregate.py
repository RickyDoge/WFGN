import torch.nn as nn
import torch.nn.functional as F
from autoencoder.encoder import AEEncoder
from autoencoder.decoder import AEDecoder


class AutoEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(AutoEncoder, self).__init__()
        self.encoder = AEEncoder(pretrained=pretrained)
        self.decoder = AEDecoder(self.encoder)
        self.loss_func = nn.HuberLoss()

    def forward(self, x):
        embedding, pool_indices = self.encoder(x)
        return self.decoder(embedding, pool_indices)

    def encode(self, x):
        return F.avg_pool2d(self.encoder(x)[0], kernel_size=(4, 4)).flatten()  # 下采样展平输出embedding

    def train_step(self, batchOfX, batchOfY):
        predict = self.forward(batchOfX)
        loss = self.loss_func(batchOfY, predict)
        return loss
