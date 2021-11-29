import os
import torch
import numpy as np
import torch.utils.data as tud
from PIL import Image
from torchvision import transforms


class ImageClusteringDataset(tud.Dataset):
    def __init__(self, pos_dir, neg_dir):
        super(ImageClusteringDataset, self).__init__()
        self.positive_samples = [os.path.join(pos_dir, fname) for fname in os.listdir(pos_dir)]
        self.negative_samples = [os.path.join(neg_dir, fname) for fname in os.listdir(neg_dir)]
        self.transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))])

    def __len__(self):  # 对于每个正样本。每次随机采样1个正样本+4个负样本
        return len(self.positive_samples) * 5

    def __getitem__(self, idx):
        # return: 3 * 256 * 256 * 2
        if idx < len(self.positive_samples):
            idx2 = np.random.randint(0, len(self.positive_samples))
            comparison = self.transform(Image.open(self.positive_samples[idx2]).convert('RGB')).unsqueeze(dim=-1)
            label = 1
        else:
            idx2 = np.random.randint(len(self.positive_samples), self.__len__())
            comparison = self.transform(Image.open(self.negative_samples[idx2]).convert('RGB')).unsqueeze(dim=-1)
            label = -1

        pivot = self.transform(Image.open(self.positive_samples[idx % len(self.positive_samples)]).convert('RGB')).unsqueeze(dim=-1)
        return torch.concat([pivot, comparison], dim=-1), label
