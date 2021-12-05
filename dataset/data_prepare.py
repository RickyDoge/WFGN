import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
from typing import List
from tqdm import tqdm
from two_tower_model.tower import TwoTowerModel
from two_tower_model.selectivesearch import selective_search, show_bbox


def adaptive_resize(batchOfImg: List[torch.Tensor]):
    img_sizes = [[image.shape[-2], image.shape[-1]] for image in batchOfImg]
    max_size = max([max(img_size[0], img_size[1]) for img_size in img_sizes])  # the maximal size
    batch_shape = (len(batchOfImg), batchOfImg[0].shape[0], max_size, max_size)  # wanted shape

    padded_images = batchOfImg[0].new_full(batch_shape, 0.0)
    for padded_img, img in zip(padded_images, batchOfImg):
        h, w = img.shape[1:]
        padded_img[..., :h, :w].copy_(img)

    return padded_images


def resized_img(in_dirs: List[str], out_dir, size):
    # resize images to size (256, 256)
    transform = transforms.Compose([transforms.Resize(size=size)])
    for in_dir in in_dirs:
        files = os.listdir(in_dir)
        bar = tqdm(files)
        bar.set_description(in_dir)
        for file in bar:
            img = Image.open(os.path.join(in_dir, file)).convert('RGB')
            img = transform(img)
            img.save(os.path.join(out_dir, file))


def random_crop(in_dirs: List[str], out_dir, repeat=1):
    transform = transforms.Compose([transforms.Resize(size=(1280, 1280)),
                                    transforms.RandomCrop(size=(256, 256))])
    for in_dir in in_dirs:
        files = os.listdir(in_dir)
        bar = tqdm(files)
        bar.set_description(in_dir)
        for file in bar:
            img = Image.open(os.path.join(in_dir, file)).convert('RGB')
            for i in range(repeat):
                img2 = transform(img)
                img2.save(os.path.join(out_dir, file.replace('.', '-{}.'.format(i + 1))))


def tSNE_visualize(gpu_accelerate=True):
    device = torch.device('cuda:0' if gpu_accelerate and torch.cuda.is_available() else 'cpu')
    model = TwoTowerModel(model_dir=None).to(device)
    transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                    transforms.ToTensor()])
    model.load_state_dict(torch.load(r'D:\Project\PyCharmProjects\WFGN\weight\tower.pth'))
    pos_dir = r'D:\Training Dataset\FurGenTMP\positive'
    neg_dir = r'D:\Training Dataset\FurGenTMP\negative'

    embeddings = []
    pos_count, neg_count = 0, 0
    with torch.no_grad():
        for img in tqdm(os.listdir(pos_dir)):
            image_tensor = transform(Image.open(os.path.join(pos_dir, img)).convert('RGB')).unsqueeze(dim=0).to(device)
            embedding = model(image_tensor).squeeze(dim=0)
            embeddings.append(embedding.cpu().numpy())
            pos_count += 1

        positive_mean = torch.tensor(np.array(embeddings))
        positive_mean = torch.mean(positive_mean, dim=0)
        print(positive_mean.shape)
        torch.save(positive_mean, r'D:\Project\PyCharmProjects\WFGN\weight\positive_mean.pth')

        for img in tqdm(os.listdir(neg_dir)):
            image_tensor = transform(Image.open(os.path.join(neg_dir, img)).convert('RGB')).unsqueeze(dim=0).to(device)
            embedding = model(image_tensor).squeeze(dim=0)
            embeddings.append(embedding.cpu().numpy())
            neg_count += 1

    tsne = TSNE(n_components=2)
    xy = tsne.fit_transform(embeddings)
    x = xy[:, 0]
    y = xy[:, 1]
    c = [0 for _ in range(pos_count)] + [1 for _ in range(neg_count)]
    plt.scatter(x, y, s=1, c=c)
    plt.show()


if __name__ == '__main__':
    '''
    resized_img(in_dirs=[r'D:\Training Dataset\FurGenTMP\新建文件夹'],
                out_dir=r'D:\Training Dataset\FurGenTMP', size=1200)
    '''
    '''
    random_crop(in_dirs=['D:\\Training Dataset\\FurGen\\2'],
                out_dir='D:\\Training Dataset\\FurGenTMP\\2333', repeat=2)
    '''
    # show_bbox(Image.open(r'D:\Training Dataset\FurGenTMP\新建文件夹\yao.jpg').convert('RGB'))
