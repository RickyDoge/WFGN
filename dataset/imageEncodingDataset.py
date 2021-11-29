import os
import torch
import torch.utils.data as tud
from PIL import Image
from torchvision import transforms


class ImageEncodingDataset(tud.Dataset):
    def __init__(self, in_dirs):
        super(ImageEncodingDataset, self).__init__()
        self.files = []
        self.transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))])
        fdirs = [os.path.join(in_dirs, fdir) for fdir in os.listdir(in_dirs)]
        for fdir in fdirs:
            self.files += [os.path.join(fdir, img_file) for img_file in os.listdir(fdir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        return self.transform(image)
