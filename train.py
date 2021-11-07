import time
from models.vit import ViT
from models.resnet_4ch import resnet18
from easydict import EasyDict
import yaml
from utils import *
import torch.nn as nn
import torchvision

experiment_model = "resnet18"
# experiment_model = "VIT"
# experiment_model = "resnet18_VIT"
CONTINUE = False
global last_epoch, best_prec, config


# 训练
def train(train_loader, net, criterion, optimizer, epoch, device):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print("===  Epoch: [{}/{}]  === ".format(epoch + 1, config.epochs))
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_index + 1) % 100 == 0:
            print("===  step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}  ===".format(
                batch_index + 1, len(train_loader), train_loss / (batch_index + 1), 100.0 * correct / total,
                get_current_lr(optimizer)))
    print("===  step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}  ===".format(
        batch_index + 1, len(train_loader), train_loss / (batch_index + 1), 100.0 * correct / total,
        get_current_lr(optimizer)))

    end = time.time()
    print("===  cost time: {:.4f}s  ===".format(end - start))


# 测试
def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    print("===  Validate [{}/{}] ===".format(epoch + 1, config.epochs))
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("===  test loss: {:.3f} | test acc: {:6.3f}%  ===".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))

    # 保存检查点
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, "experiment/" + experiment_model + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


if __name__ == "__main__":
    print("experiment_model：" + experiment_model)
    # 打开配置文件
    with open("experiment/" + experiment_model + '/config.yaml') as f:
        config = yaml.load(f)
    config = EasyDict(config)

    # 选择ResNet18
    if experiment_model == 'resnet18':
        net = resnet18(pretrained=config.pretrained)
        print(config.pretrained)
        # 改变resnet的最后一个线性层的输出
        channel_in = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(channel_in, config.num_classes))

    # 选择VIT
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

    # 选择resnet18+vit
    else:
        backbone = resnet18(pretrained=config.pretrained)
        # 去掉池化层及全连接层, resnet18 layer4 output shape: b,512,8,8
        features = list(backbone.children())[:-2]
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

    # 训练设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), config.lr_scheduler.base_lr)
    # 继续上一次训练
    if CONTINUE:
        ckpt_file_name = experiment_model + '/' + config.ckpt_name + '.pth.tar'
        best_prec, last_epoch, optimizer = load_checkpoint(ckpt_file_name, net, optimizer=optimizer)
    else:
        last_epoch = -1
        best_prec = 0

    # 加载训练数据
    transform1 = transforms.Compose([transforms.Resize((config.input_size, config.input_size))])
    transform2 = transforms.Compose([transforms.ToTensor()])
    train_loader, test_loader = get_data_loader(transform1, transform2, config)

    # 训练
    print(("=======  Training  ======="))
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        start_train_8batchsize = time.perf_counter()
        train(train_loader, net, criterion, optimizer, epoch, device)
        end_train_8batchsize = time.perf_counter()
        print("1 epoch of 8 batchsize run time is : %f seconds" % (end_train_8batchsize - start_train_8batchsize))
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    print(("=======  Training Finished.Best_test_acc: {:.3f}% ========".format(best_prec)))
