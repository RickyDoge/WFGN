import os
import torch
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.to_tensor = transforms.ToTensor()
        self.to_PIL = transforms.ToPILImage()
        self.threshold = threshold

    def merge_bbox(self, bboxs, t=0.6):
        # 思路类似于NMS算法(Non-Maximum Suppression)，但是目的是为了扩充
        # bboxs: list of (confidence, (x,y,w,h))。候选框的置信度，候选框的xy坐标和长宽

        def iou(box1, box2):
            in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
            in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
            inner = 0 if in_h < 0 or in_w < 0 else in_h * in_w
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inner
            iou = inner / union
            return iou

        def merge(box1, box2):
            x0, y0, w0, h0 = box1
            x1, y1, w1, h1 = box2
            x, y = min(x0, x1), min(y0, y1)
            if x == x0 and y == y0:
                return x, y, max(w1 + abs(x0 - x1), w0), max(h1 + abs(y0 - y1), h0)
            elif x == x1 and y == y1:
                return x, y, max(w0 + abs(x0 - x1), w1), max(h0 + abs(y0 - y1), h1)
            elif x == x0:
                return x, y, max(x1 - x + w1, w0), max(y0 - y + h0, h1)
            else:
                return x, y, max(x0 - x + w0, w1), max(y1 - y + h1, h0)

        bboxs.sort(reverse=True)
        for box in reversed(bboxs[1:]):
            c, (x0, y0, w0, h0) = bboxs[0]
            _, (x1, y1, w1, h1) = box
            if iou((x0, y0, x0 + w0, y0 + h0), (x1, y1, x1 + w1, y1 + h1)) > t:
                if 1.5 > w0 * h0 / w1 / h1 > 0.66:
                    bboxs[0] = (c, merge(bboxs[0][1], box[1]))
        return bboxs[0]

    def output(self, out_dir, k=20):
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                img_loader = self.dataset[i]
                target, target_fdir = None, ''

                heap = []  # 小根堆
                max_c = self.threshold
                for bbox, coor, fdir in img_loader:
                    img = self.transform(bbox).to(self.device)
                    embedding = self.model(img.unsqueeze(dim=0)).squeeze(dim=0)
                    confidence = torch.dot(embedding, self.positive_mean).item()
                    if confidence > max_c:
                        # max_c = confidence
                        target, target_fdir = bbox, fdir
                        if len(heap) < k:
                            heapq.heappush(heap, (confidence, coor))
                        else:
                            heapq.heappushpop(heap, (confidence, coor))

                if target_fdir != '':
                    _, (x, y, w, h) = self.merge_bbox(heap)
                    ori_img = self.to_tensor(Image.open(fdir).convert('RGB'))
                    ori_img = ori_img[:, max(0, y-h//3): int(y + h*6/5), max(0, x-w//3): int(x + w*4/3)]  # 微调
                    # Linux等系统下，用/。Windows系统下，用\\
                    self.to_PIL(ori_img).save(os.path.join(out_dir, target_fdir.split('\\')[-1]))


if __name__ == '__main__':
    '''
    handler = ImageEvalStream(gpu_accelerate=False)
    handler.plot_img(handler.transform(Image.open('/content/drive/My Drive/test_img.png').convert('RGB')),
                     label='original pic')
    handler.plot_img(handler.autoencode_img('/content/drive/My Drive/test_img.png'), label='encoded pic')
    '''
    detection = ObjectDetectionStream(in_dir=r'D:\Training Dataset\FurGen\original',
                                      tensor_dir=r'D:\Project\PyCharmProjects\WFGN\weight\positive_mean.pth',
                                      model_dir=r'D:\Project\PyCharmProjects\WFGN\weight\tower.pth',
                                      threshold=0)
    detection.output(out_dir=r'D:\Training Dataset\FurGen\genHead')
