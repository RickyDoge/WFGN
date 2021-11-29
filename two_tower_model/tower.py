import os
import torch
import torch.nn.functional as F
from torch import nn
from autoencoder.encoder import AEEncoder


class TwoTowerModel(nn.Module):
    def __init__(self, model_dir=os.path.join(os.curdir, 'weight', 'autoencoder.pth')):
        super(TwoTowerModel, self).__init__()
        self.tower = AEEncoder()
        if model_dir is not None:
            tmp = dict()
            model_state = torch.load(model_dir)
            for parameter in model_state:
                hierarchy = parameter.split('.')
                if hierarchy[0] == 'encoder':
                    tmp['.'.join(hierarchy[1:])] = model_state[parameter]

            self.tower.load_state_dict(tmp)
        self.loss_func = nn.CosineEmbeddingLoss(margin=0.1)  # margin=0.1，允许一定范围内的“不相似度”

    def forward(self, img):
        feature_map, _ = self.tower(img)
        return F.avg_pool2d(feature_map, kernel_size=(4, 4)).flatten(start_dim=1)  # 2048 dimensional embedding

    def train_step(self, batch_input, label):  # label为1表示相关，-1表示不相关
        feat1 = self.forward(batch_input[:, :, :, :, 0].squeeze(-1)).squeeze(dim=-1)
        feat2 = self.forward(batch_input[:, :, :, :, 1].squeeze(-1)).squeeze(dim=-1)
        return self.loss_func(feat1, feat2, label)
