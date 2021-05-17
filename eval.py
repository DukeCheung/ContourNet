import os

import cv2
import torch
from config import Config
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from dataloader import ImageDataset
import numpy as np
import torch.nn as nn

args = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

if __name__ == '__main__':
    device = torch.device('cuda')

    model = models.resnet50(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 3)
    model.to(device)
    # print(model)
    model.load_state_dict(torch.load(args.param_file))
    model.eval()

    min_loss = float('inf')

    dataset = ImageDataset(args.file_path)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_workers,
                            drop_last=args.drop_last)

    test_img = cv2.imread('/home/zhangxing/PycharmProjects/ContourNet/test.jpg')
    test_img = np.float32(cv2.resize(test_img, (224, 224))) / 255

    test_img = np.transpose(test_img, [2, 1, 0])
    test_img = torch.from_numpy(test_img).to(device)
    test_img = test_img.unsqueeze(dim=0)

    label = model(test_img)

    i = 0
    min_index = -1
    min_loss = float('inf')

    for img in dataset.images:
        img = np.transpose(img, [2, 1, 0])
        img = img.unsqueeze(dim=0)
        img = img.to(device)

        predict = model(img)
        loss = F.pairwise_distance(predict, label, p=2).sum()

        if min_loss > loss.item():
            min_loss = loss.item()
            min_index = i

        i += 1
    print('min loss is {}, index is {}, filename is {}'.format(min_loss, min_index, dataset.file_name[min_index]))