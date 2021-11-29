import os
import torch
import torch.utils.data as tud
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from autoencoder.aggregate import AutoEncoder
from two_tower_model.tower import TwoTowerModel
from dataset.imageEncodingDataset import ImageEncodingDataset
from dataset.imageClusteringDataset import ImageClusteringDataset


def train_autoencoder(num_epochs=30, lr=3e-4, batch_size=24, decay=0, scheduler_step=5, scheduler_gamma=0.5):
    dataset = ImageEncodingDataset('D:\\Training Dataset\\FurGen')
    dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    optimizer = Adam([{'params': model.encoder.parameters(), 'lr': lr * 0.1},  # 迁移学习encoder，设置较小学习率
                      {'params': model.decoder.parameters(), 'lr': lr}], weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    min_loss = float('inf')
    checkpoint_epoch = -1

    if os.path.isfile(os.path.join(os.curdir, 'weight', 'fursonaGenerator.checkpoint')):  # 加载存档点
        checkpoint = torch.load(os.path.join(os.curdir, 'weight', 'fursonaGenerator.checkpoint'))
        checkpoint_epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        min_loss = checkpoint['min_loss']
        print('load checkpoint at epoch {}, min_loss {:.4f}'.format(checkpoint_epoch, min_loss))

    for epoch in range(num_epochs):
        if epoch <= checkpoint_epoch:
            continue

        ep_loss = 0
        num_it = 0
        bar = tqdm(iter(dataloader))
        for it, x in enumerate(bar):
            x_gpu = x.to(device)
            optimizer.zero_grad()
            loss = model.train_step(x_gpu, x_gpu)
            ep_loss += loss.item()
            num_it += 1
            loss.backward()
            optimizer.step()
            if it % 10 == 1:
                bar.set_description('Epoch {}, It {}, Loss {:.4f}'.format(epoch, it, ep_loss / num_it))
        scheduler.step()

        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'min_loss': min_loss}, os.path.join(os.curdir, 'weight', 'fursonaGenerator.checkpoint'))

        ep_loss = ep_loss / num_it
        if ep_loss < min_loss:
            min_loss = ep_loss
            torch.save(model.state_dict(), os.path.join(os.curdir, 'weight', 'autoencoder.pth'))
        elif (ep_loss - min_loss) / min_loss > 0.02:  # early stop
            break


def train_tower(num_epochs=30, lr=1e-4, batch_size=24, decay=0, scheduler_step=5, scheduler_gamma=0.5):
    dataset = ImageClusteringDataset(pos_dir=r'D:\Training Dataset\FurGenTMP\positive',
                                     neg_dir=r'D:\Training Dataset\FurGenTMP\negative')
    dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    min_loss = float('inf')
    checkpoint_epoch = -1
    print(model)
    if os.path.isfile(os.path.join(os.curdir, 'weight', 'TwoTower.checkpoint')):  # 加载存档点
        checkpoint = torch.load(os.path.join(os.curdir, 'weight', 'TwoTower.checkpoint'))
        checkpoint_epoch = checkpoint['epoch']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        min_loss = checkpoint['min_loss']
        print('load checkpoint at epoch {}, min_loss {:.4f}'.format(checkpoint_epoch, min_loss))

    for epoch in range(num_epochs):
        if epoch <= checkpoint_epoch:
            continue

        ep_loss = 0
        num_it = 0
        bar = tqdm(iter(dataloader))
        for it, (x, y) in enumerate(bar):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            optimizer.zero_grad()
            loss = model.train_step(x_gpu, y_gpu)
            ep_loss += loss.item()
            num_it += 1
            loss.backward()
            optimizer.step()
            if it % 10 == 1:
                bar.set_description('Epoch {}, It {}, Loss {:.4f}'.format(epoch, it, ep_loss / num_it))
        scheduler.step()

        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'min_loss': min_loss}, os.path.join(os.curdir, 'weight', 'TwoTower.checkpoint'))

        ep_loss = ep_loss / num_it
        if ep_loss < min_loss:
            min_loss = ep_loss
            torch.save(model.state_dict(), os.path.join(os.curdir, 'weight', 'autoencoder.pth'))
        elif (ep_loss - min_loss) / min_loss > 0.02:  # early stop
            break


if __name__ == '__main__':
    # train_autoencoder()
    train_tower()
