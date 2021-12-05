import os
from PIL import Image
from torchvision import transforms
from two_tower_model.selectivesearch import extract_bbox


class ObjectDetectionDataset(object):
    def __init__(self, in_dir):
        self.transform = transforms.Compose([
            transforms.Resize(1200),
            transforms.ToTensor()
        ])
        self.files = [os.path.join(in_dir, fname) for fname in os.listdir(in_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):  # 返回该图片的所有bbox构成的列表的迭代器
        fdir = self.files[idx]
        img = self.transform(Image.open(fdir).convert('RGB'))
        bboxs = extract_bbox(img.transpose(0, 2).transpose(0, 1))
        return self.construct_iterator([(img[:, y:y + h, x:x + w], (x, y, w, h), fdir) for (x, y, w, h) in bboxs])

    def construct_iterator(self, candidate_imgs):  # 构造迭代器
        for img, (x, y, w, h), fdir in candidate_imgs:
            yield img, (x, y, w, h), fdir
