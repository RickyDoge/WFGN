import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from autoencoder.aggregate import AutoEncoder
from two_tower_model.tower import TwoTowerModel
from dataset.objectDetectionDataset import ObjectDetectionDataset


class ImageEvalStream():
    def __init__(self, gpu_accelerate=True):
        self.transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))])
        self.denorm = transforms.Compose([transforms.Normalize((-0.4914 / 0.229, -0.4822 / 0.224, -0.44651 / 0.225),
                                                               (1 / 0.229, 1 / 0.224, 1 / 0.225)),
                                          transforms.ToPILImage()])
        self.device = torch.device('cuda:0' if gpu_accelerate and torch.cuda.is_available() else 'cpu')
        self.autoencoder = AutoEncoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load(os.path.join(os.curdir, 'weight', 'autoencoder.pth')))
        self.autoencoder.eval()

    def autoencode_img(self, img_file: str):
        with torch.no_grad():
            image = self.transform(Image.open(img_file).convert('RGB')).to(self.device)
            return self.autoencoder(image.unsqueeze(0)).squeeze(0)

    def encode_img(self, img_file: str):
        with torch.no_grad():
            image = self.transform(Image.open(img_file).convert('RGB')).to(self.device)
            return self.autoencoder.encode(image.unsqueeze(0)).squeeze(0)

    def encode_imgs(self, imgs_fdir: str):
        imgs_embedding = np.array([self.encode_img(os.path.join(imgs_fdir, img_file)).numpy() for img_file in
                                   tqdm(os.listdir(imgs_fdir))])
        dataFrame = pd.DataFrame(imgs_embedding, index=list(os.listdir(imgs_fdir)))
        return dataFrame

    def plot_img(self, img, label=''):
        visible = self.denorm(img)
        plt.imshow(visible)
        plt.title(label=label)
        plt.show()
        plt.clf()


class ObjectDetectionStream():
    def __init__(self, in_dir, tensor_dir, model_dir, threshold=0.2, gpu_accelerate=True):
        self.device = torch.device('cuda:0' if gpu_accelerate and torch.cuda.is_available() else 'cpu')
        self.dataset = ObjectDetectionDataset(in_dir=in_dir)
        self.positive_mean = torch.load(tensor_dir).to(self.device)
        self.model = TwoTowerModel().to(self.device)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225))])
        self.denorm = transforms.Compose([transforms.Normalize((-0.4914 / 0.229, -0.4822 / 0.224, -0.44651 / 0.225),
                                                               (1 / 0.229, 1 / 0.224, 1 / 0.225)),
                                          transforms.ToPILImage()])
        self.to_PIL = transforms.ToPILImage()
        self.threshold = threshold

    def output(self, out_dir):
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                img_loader = self.dataset[i]
                max_sim = self.threshold
                target = None
                target_coor = None
                target_fdir = ''
                for bbox, coor, fdir in img_loader:
                    img = self.transform(bbox).to(self.device)
                    embedding = self.model(img.unsqueeze(dim=0)).squeeze(dim=0)
                    confidence = torch.dot(embedding, self.positive_mean)
                    if confidence > max_sim:
                        max_sim = confidence
                        target = bbox
                        target_coor = coor
                        target_fdir = fdir
                if target is not None:
                    if out_dir is None:  # 展示结果
                        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                        ori_img = Image.open(fdir).convert('RGB')
                        ax.imshow(ori_img)
                        x, y, w, h = target_coor
                        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3)
                        ax.add_patch(rect)
                        plt.show()
                    else:  # 保存结果
                        # Linux等系统下，\\可能需要换成/。Windows系统下，用\\
                        self.to_PIL(target).save(os.path.join(out_dir, target_fdir.split('\\')[-1]))
                        # print('\nImg No.{}. Confidence {:.4f}'.format(i, max_sim))


if __name__ == '__main__':
    '''
    handler = ImageEvalStream(gpu_accelerate=False)
    handler.plot_img(handler.transform(Image.open('/content/drive/My Drive/test_img.png').convert('RGB')),
                     label='original pic')
    handler.plot_img(handler.autoencode_img('/content/drive/My Drive/test_img.png'), label='encoded pic')
    '''
    detection = ObjectDetectionStream(in_dir=r'D:\Training Dataset\FurGenTMP\新建文件夹',
                                      tensor_dir=r'D:\Project\PyCharmProjects\ImagePlay\weight\positive_mean.pth',
                                      model_dir=r'D:\Project\PyCharmProjects\ImagePlay\weight\tower.pth',
                                      threshold=-100)
    detection.output(out_dir=None)
