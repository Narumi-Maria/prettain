import os
import yaml
from easydict import EasyDict
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from models.resnet_4ch import resnet18
from models.vit import ViT
import numpy as np
from utils import *

experiment_model = "resnet18"
# experiment_model = "VIT"
# experiment_model = "resnet18_VIT"

global config

if __name__ == '__main__':

    # 加载配置文件
    with open("experiment/" + experiment_model + '/config.yaml') as f:
        file_data = f.read()
        config = yaml.load(file_data)
    config = EasyDict(config)
    print("load config")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 选择模型
    if experiment_model == 'resnet18':
        net = resnet18(pretrained=False)  # pretrained 是否使用预训练模型
        net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))
        ckpts_path = 'experiment/' + experiment_model + '/Net.pth.tar'

    elif experiment_model == 'VIT':
        net = ViT(backbone=None,
                  image_size=256,
                  patch_size=32,
                  num_classes=config.num_classes,
                  dim=1024,
                  depth=6,
                  heads=16,
                  mlp_dim=2048,
                  dropout=0.1,
                  emb_dropout=0.1
                  )
        ckpts_path = 'experiment/' + experiment_model + '/Net.pth.tar'

    else:
        backbone = resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        features = list(backbone.children())[:-2]  # 去掉池化层及全连接层, resnet18 layer4 output shape: b,512,8,8
        backbone = nn.Sequential(*features)

        net = ViT(backbone=backbone,
                  image_size=8,
                  patch_size=2,
                  channels=512,
                  num_classes=config.num_classes,
                  dim=1024,
                  depth=6,
                  heads=16,
                  mlp_dim=2048,
                  dropout=0.1,
                  emb_dropout=0.1
                  )
        ckpts_path = 'experiment/' + experiment_model + '/Net.pth.tar'

    # 加载权重
    net_dict = net.state_dict()
    state_dict = torch.load(ckpts_path)['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net_dict.update(new_state_dict)
    net.load_state_dict(net_dict)
    net.to(device)
    net.eval()

    # 加载训练数据
    transform1 = transforms.Compose([transforms.Resize((config.input_size, config.input_size))])
    transform2 = transforms.Compose([transforms.ToTensor()])
    train_loader, test_loader = get_data_loader(transform1, transform2, config)

    # 预测
    preds = []
    gts = []
    with torch.no_grad():
        for batch_index, (inputs_batch, targets_batch) in enumerate(test_loader):
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            outputs = net(inputs_batch)

            preds.extend(outputs.max(1)[1].cpu().numpy())
            gts.extend(targets_batch.cpu().numpy())

    # 计算F1和Balanced accuracy
    tp = sum(list(map(lambda a, b: a == 1 and b == 1, preds, gts)))
    fp = sum(list(map(lambda a, b: a == 1 and b == 0, preds, gts)))
    fn = sum(list(map(lambda a, b: a == 0 and b == 1, preds, gts)))
    tn = sum(list(map(lambda a, b: a == 0 and b == 0, preds, gts)))
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    bal_acc = (tpr + tnr) / 2
    print(experiment_model + " : f1={:.3f},bal_acc={:.3f}".format(f1, bal_acc))
