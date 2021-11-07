import os
import math
import shutil
import numpy as np
from PIL import Image
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


# 数据集处理
class ImageDataset(Dataset):
    def __init__(self, images_dir, mask_dir, txt_path, img_transform1=None, img_transform2=None, istrain=True):
        with open(txt_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]

        self.labels = []
        self.images_path = []
        self.mask_path = []

        for row in reader:
            label = int(row[-3])
            image_path = row[-2]
            mask_path = row[-1]

            self.labels.append(label)
            #self.images_path.append(os.path.join(images_dir, image_path))
            self.images_path.append(image_path)
            #self.mask_path.append(os.path.join(mask_dir, mask_path))
            self.mask_path.append(mask_path)

        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
        self.istrain = istrain

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')
        mask = Image.open(self.mask_path[index]).convert('L')  # gray
        label = self.labels[index]

        img = self.img_transform1(img)
        mask = self.img_transform1(mask)
        # 数据增强，合成图和mask要同时翻转
        transforms_flap = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        if self.istrain is True and np.random.uniform() < 0.5:
            img = transforms_flap(img)
            mask = transforms_flap(mask)
        img = self.img_transform2(img)
        mask = self.img_transform2(mask)

        img_cat = torch.cat((img, mask), dim=0)


        return img_cat, label

    def __len__(self):
        return len(self.labels)


# 加载checkpoint
def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        print("=== loading checkpoint '{}' ===".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=== done. also loaded optimizer from epoch {}) ===".format(last_epoch + 1))
            return best_prec, last_epoch, optimizer


# 保存checkpoint
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


# 加载训练数据
def get_data_loader(transform1, transform2, config):
    trainset = ImageDataset(images_dir=config.images_dir, mask_dir=config.mask_dir, txt_path=config.train_txt,
                            img_transform1=transform1, img_transform2=transform2, istrain=True)
    testset = ImageDataset(images_dir=config.images_dir, mask_dir=config.mask_dir, txt_path=config.val_txt,
                           img_transform1=transform1, img_transform2=transform2, istrain=False)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                               shuffle=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch,
                                              shuffle=False, num_workers=config.workers)
    return train_loader, test_loader


# 得到当前学习率
def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 调整学习率
def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
             (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
             (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
             (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
             (1.0 - math.tanh(
                 config.lr_scheduler.lower_bound
                 + (config.lr_scheduler.upper_bound
                    - config.lr_scheduler.lower_bound)
                 * ratio)
              ) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    from models.resnet_4ch import resnet18
    from easydict import EasyDict
    import yaml

    net = resnet18()
    optimizer = torch.optim.Adam(net.parameters(), 0.0001)
    with open(r"experiment/resnet18/config.yaml") as f:
        config = yaml.load(f)
    config = EasyDict(config)
    lr = adjust_learning_rate(optimizer, 3, config)
    print("sjah")
