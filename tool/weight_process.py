import os
import sys
import torch


class WeightTool:
    def __init__(self):
        pass

    # ---------------------保存权重---------------------#
    @staticmethod
    def save_weight(Model, loss, epoch, save_path, save_file_name):
        ckpt = dict()
        ckpt["state_dict"] = Model.state_dict()
        ckpt["best_loss"] = loss
        ckpt["best_epoch"] = epoch
        torch.save(ckpt, os.path.join(save_path, save_file_name))
        # print(f"{save_path},{save_file_name}模型已经保存")

    # ---------------------加载权重---------------------#
    @staticmethod
    def load_weight(weight_path, file_name):
        ckpt = torch.load(os.path.join(weight_path, file_name))  # 加载保存模型的字典tar文件
        loss_load = ckpt["best_loss"]
        epoch_load = ckpt["best_epoch"]

        # 加载模型文件
        weight_load = ckpt["state_dict"]
        print(f'Prev Best_loss:{loss_load}, Prev Best_epoch:{epoch_load}')
        print('模型开始测试...')
        return weight_load
