import mediapipe as mp
import os
import torch
import json
from torchvision import transforms
import cv2 as cv
import numpy as np
from PIL import Image
from tool.Gesture import Mphand
from tool.data_process import DataProcess as Dp
from tool.weight_process import WeightTool as Wt
from model.MobileV2 import MobileNetV2

weight_source = "./weight"
# weight_name = "Gesture_train_lr.tar"  # 动态调整学习率的权重
weight_name = "Gesture_train.tar"
img_source = "./img_ceshi"
json_path = './class_gender.json'

assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r", encoding='gbk') as f:
    class_indict = json.load(f)


def main():
    class_text = 'domada'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_name = input('请输入预测图片的名称(加后缀):')  # 图片必须在img_ceshi文件夹内
    image_file = os.path.join(img_source, img_name)
    image = Image.open(image_file)

    # --------------------图像预处理模块----------------------#
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # [N, C, H, W]
    image_tensor = data_transform(image)
    # expand batch dimension 添加一个batch维度
    image_tensor = torch.unsqueeze(image_tensor, dim=0)

    # --------------------分类模块----------------------#
    net = MobileNetV2(num_classes=2)
    net = net.to(device)
    weight_load = Wt.load_weight(weight_source, weight_name)
    net.load_state_dict(weight_load, strict=False)
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(image_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        tempt = 0
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
        if predict[i].numpy() > tempt:
            class_text = " Class: " + class_indict[str(i)] + " Prob: " + str(predict[i].numpy())
            tempt = predict[i].numpy()

    # --------------------识别模块----------------------#
    img_Dp = Dp(image, [512, 512])
    new_image, _, _ = img_Dp.letterbox_image()  # _ 为占位符
    # new_image.show()  # 查看裁剪效果

    new_image = cv.cvtColor(np.asarray(new_image), cv.COLOR_BGR2RGB)  # PIL转opencv
    # image = Image.fromarray(cv2.cvtColor(img_ceshi, cv2.COLOR_BGR2RGB))  # opencv转PIL

    mp = Mphand(new_image)
    mp.Select_Mode(class_text, mode=True)  # mode=True(default)：姿态估计， False：手势识别


if __name__ == '__main__':
    main()
