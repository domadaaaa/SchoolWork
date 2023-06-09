import cv2
import mediapipe as mp
import time
from model.MobileV2 import MobileNetV2
from model.Resnet50 import Res50
from torchvision import models
import torch
import numpy as np
import json
import os
from PIL import Image
from tool.weight_process import WeightTool as Wt
from torchvision import transforms

# ----------------------↓ 设置模式---------------------------#
mode = False  # mode=True(default)：姿态估计 | False：手势识别
# ------------------↑ 请先设置模式再运行！！-------------------#
num_classes = 2  # 分类的类别

json_path = './class_gender.json'
weight_source = "./weight"
weight_name = "Gesture_acc_mob.tar"  # MobileNet
# weight_name = "Gesture_acc_res.tar"  # ResNet
video_path = "./video_ceshi"

# video_name = input('请输入检测视频的名称(加后缀):')  # 视频必须在video_ceshi文件夹内
# video_file = os.path.join(video_path, video_name)


with open(json_path, "r", encoding='gbk') as f:
    class_indict = json.load(f)


# ---------------------MobileNet-------------------------#
net = MobileNetV2(num_classes)
weight_load = Wt.load_weight(weight_source, weight_name)
# -------------------------------------------------------#

# ---------------------ResNet50--------------------------#
# net = models.resnet50()
# num_ftrs = net.fc.in_features
# net.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, num_classes),
#                        torch.nn.LogSoftmax(dim=1))
# weight_load = Wt.load_weight(weight_source, weight_name)
# -------------------------------------------------------#

# 模型预测相关
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device.type == "cuda":
    net = net.to(device)
    print("using GPU")
else:
    print("using CPU")
net.load_state_dict(weight_load, strict=False)
net.eval()
class_text = ""

# 初始化 MediaPipe 中的手部检测和跟踪模型
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=True,
                   model_complexity=1,
                   smooth_landmarks=True,
                   # enable_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)


# 设置变量以跟踪帧率
pTime = 0
cTime = 0

# 进入视频捕获循环
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # numpy.ndarray  用于mp输入
    img_p = Image.fromarray(np.uint8(imgRGB))  # np->PIL 用于网络输入(CenterCrop只接受PIL图像)

    # --------------------图像预处理模块----------------------#
    data_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_tensor = data_transform(img_p)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    # 预测模块
    with torch.no_grad():
        if device.type == "cuda":
            output = torch.squeeze(net(image_tensor.to(device))).cpu()  # GPU推理
        else:
            output = torch.squeeze(net(image_tensor)).cpu()  # CPU推理
        predict = torch.softmax(output, dim=0)
        tempt = 0.0
    for i in range(len(predict)):
        if predict[i].numpy() > tempt:
            class_text = " Gender: " + class_indict[str(i)] + " Prob: " + str(predict[i].numpy())
            tempt = predict[i].numpy()
    # ----------------------------------------------姿态-----------------------------------------------#
    if mode:
        results0 = pose.process(imgRGB)
        mpDraw.draw_landmarks(img, results0.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # ----------------------------------------------手势-----------------------------------------------#
    else:
        results = hands.process(imgRGB)
        # 对每一帧进行手部检测和关键点提取，并在图像上绘制关键点和连接线
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # id代表的是关节点的个数
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # 节点用实心圆表示
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # 计算并显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # 输出信息
    cv2.putText(img, "fps=" + str(int(fps)) + "|" + class_text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 255, 0), 2)

    # 显示图像并等待按键输入退出循环
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按"q"键退出
        break
