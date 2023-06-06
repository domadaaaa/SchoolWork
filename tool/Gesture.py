import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os

import numpy as np


# 传进来的是Opencv读取的图片(np.array)
class Mphand:
    def __init__(self, img):  # 要求输入的img为np.array，并且为RGB模式
        # 初始化 MediaPipe 中的手部检测和跟踪模型
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=True,
                                     model_complexity=1,
                                     smooth_landmarks=True,
                                     # enable_segmentation=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils  # 画手骨骼点
        self.img = img

    def Select_Mode(self, text, mode=True):
        img = self.img
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转换成RGB防止原图是opencv读取的图片
        if mode:  # mode==1代表检测人体姿态
            # 姿态
            result_save_path = "./results/pose/"
            results0 = self.pose.process(imgRGB)
            self.mpDraw.draw_landmarks(img, results0.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            # self.mpDraw.plot_landmarks(results0.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)  # 骨骼点云图
            # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 0), 2)
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
            cv2.imwrite(result_save_path + "save_imgp.jpg", img)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:  # 否则代表检测人手势
            # 手势
            result_save_path = "./results/hand/"
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    # id代表的是关节点的个数
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(f"关节点编号:{id}, x坐标:{cx}, y坐标:{cy}")
                        # 图片, 圆心坐标, 原的直径, 圆的颜色, 实心圆
                        cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # 画关键连接线

            # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 0), 2)

            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
            cv2.imwrite(result_save_path + "save_imgh.jpg", img)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
