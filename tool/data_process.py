import copy
import numpy as np
from torchvision.transforms import transforms
import torch
from PIL import Image
import cv2


class DataProcess:
    def __init__(self, image, inp_dim):  # inp_dim为目标的尺寸大小
        # 进行原始图片的备份
        # opencv读取图片(height,width,channel)
        self.image = image
        if isinstance(image, Image.Image):  # PIL读取
            image_array = np.array(self.image)
            self.ori_h = image_array.shape[0]  # 计算原图高
            self.ori_w = image_array.shape[1]  # 计算原图宽
        elif isinstance(image, (cv2.UMat, cv2.VideoCapture)):  # opencv读取
            self.ori_h = image.shape[0]  # 计算原图高
            self.ori_w = image.shape[1]  # 计算原图宽
        else:
            print("Unknown Image type")
        self.w = inp_dim[0]  # 412
        self.h = inp_dim[1]  # 412

    def letterbox_image(self):   # letterbox_image：letterbox_image是一种不失真的resize
        scale = min(self.w / self.ori_w, self.h / self.ori_h)
        new_w = int(self.ori_w * scale)
        new_h = int(self.ori_h * scale)
        self.image = self.image.resize((new_w, new_h), Image.BICUBIC)  # 将原始图片用Image.BICUBIC的方式进行缩放

        # 保留周围的黑边
        new_image = Image.new('RGB', (self.h, self.w), (128, 128, 128))  # 生成一张(self.h, self.w)尺寸的(128, 128, 128)色图片
        new_image.paste(self.image, ((self.w - new_w) // 2, (self.h - new_h) // 2))  # 将新的图片上, box是要粘贴到的区域(左上原点)

        # 不保留周围的黑边
        # new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
        # new_image.paste(self.image, (0, 0))

        return new_image, new_w, new_h
