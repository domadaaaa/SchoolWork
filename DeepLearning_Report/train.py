import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model.MobileV2 import MobileNetV2
from model.Resnet50 import Res50
from torchvision import models
from tool.weight_process import WeightTool as Wt
from tensorboardX import SummaryWriter  # 可视化
from torch.autograd import Variable

# 参数定义
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
batch_size = 64  # resnet
epochs = 30
num_classes = 2
log_dir = "./logger"
writer = SummaryWriter(logdir=log_dir)

# create model

# net = MobileNetV2(num_classes=num_classes)  # mobilenet

# net = Res50(num_class=2)  # self_resnet,自己写的ResNet50模型,由于没采用预训练，建议将batchsize调低

net = models.resnet50(pretrained=True)  # pytorch_resnet
num_ftrs = net.fc.in_features
for param in net.parameters():
    param.requires_grad = False  # 冻结模型的参数，
net.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                       nn.LogSoftmax(dim=1))


net.to(device)

# define loss function
loss_function = nn.CrossEntropyLoss()
# construct an optimizer
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.001)
# dynamic lr(MultiStepLR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
SavePath = './weight'
nw = 0  # number of workers

# 图片预处理
data_transform = {
    "Training": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "Validation": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# 获取数据集的相对目录
image_path = "./gender_dataset"
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Training"),
                                     transform=data_transform["Training"])
train_num = len(train_dataset)
gender_list = train_dataset.class_to_idx

cla_dict = dict((val, key) for key, val in gender_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 将python对象编码成Json字符串，indent表示缩进

with open('class_gender.json', 'w') as json_file:
    json_file.write(json_str)
print('Using {} dataloader workers every process'.format(nw))
# {'female':0 , 'male':1}

# Dataset & Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)
train_steps = len(train_loader)
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Validation"),
                                        transform=data_transform["Validation"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))

# MobileNetV2 FineTune
if isinstance(net, MobileNetV2):
    model_weight_path = "weight/mobilenet_v2.pth"  # 预训练权重
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    # delete classifier weights  删除分类头的权重
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    # freeze features weights 冻结特征提取部分的所有权重（除全连接层外所有参数），注释掉则训练整个网络的权重
    for param in net.features.parameters():
        param.requires_grad = False  # 不进行参数更新


def train():
    # train
    net.train()
    min_loss = float("inf")  # 保留每个epoch的最佳的loss
    for epoch in range(epochs):
        running_loss = 0.0  # 记录每个epoch训练的平均loss
        train_bar = tqdm(train_loader, file=sys.stdout)  # tqdm为进度条库,
        for step, data in enumerate(train_bar):  # step即batch_idx, 循环周期为一个batch_size
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if min_loss > loss.item():
                min_loss = loss.item()
                # 动态调整学习率
                Wt.save_weight(Model=net, loss=min_loss, epoch=epoch+1,
                               save_path=SavePath, save_file_name='Gesture_train_lr.tar')  # 保留loss最小的权重
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())
        writer.add_scalar('train_loss', running_loss/train_steps, epoch+1)  # 记录每个epoch的最佳loss

        # val
        weight_load = Wt.load_weight(weight_path=SavePath, file_name='Gesture_train_lr.tar')
        net.load_state_dict(weight_load, strict=False)
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        best_accurate = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_accurate:
            Wt.save_weight(Model=net, loss=min_loss, epoch=epoch+1,
                           save_path=SavePath, save_file_name='Gesture_acc_res.tar')  # 保留准确率最高的权重
        print('minloss_val_accuracy:{:.3f}'.format(val_accurate))
        writer.add_scalar('minloss_val_accuracy', val_accurate, epoch + 1)  # 记录每一轮验证集的准确率
    print('Finished Training')
    writer.close()


def main():
    train()


if __name__ == '__main__':
    main()
