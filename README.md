# Welcome  
# DeepLearning_report使用指南
测试图片可以在img_ceshi文件夹下找到,测试视频可以在video_ceshi文件夹下找到  
![](https://github.com/domadaaaa/SchoolWork/blob/master/深度学习报告代码/img_ceshi/ikun.jpg)  
# ---------图片示例---------- #
进入Terminal，cd进入DeepLearning_Report，运行以下指令：
```
pip install -r requirements.txt
```
```
python predict.py
```
```
ikun.jpg
```
  
# ---------视频示例---------- #
进入Terminal，cd进入DeepLearning_Report，运行以下指令：
```
pip install -r requirements.txt
```
```
python Video_detect.py
```
tips:  
1.视频默认为电脑摄像头，如果需要检测视频，请修改对应模式和文件路径  
2.想要训练先要下载对应的数据集到gender_dataset文件夹内（文件结构如下图所示）  
```
gender_dataset  
  ---Training
    ------A
    ------B
  ---Validation
    ------A
    ------B
```

