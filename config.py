import os

class Config:
    def __init__(self):
        self.file_path = '/home/zhangxing/PycharmProjects/ContourNet/dataset/'
        self.learning_rate = 0.01
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = False
        self.epochs = 5000
        self.weight_decay = 0
        self.num_workers = 0
        self.param_file = '/home/zhangxing/PycharmProjects/ContourNet/net.pth'
