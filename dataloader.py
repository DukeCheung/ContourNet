import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, file_path):
        super(Dataset, self).__init__()
        self.images = []
        self.labels = []
        for root, sub_dir, files in os.walk(file_path):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                self.images.append(torch.from_numpy(img))
                label = np.zeros(1000, dtype=np.float32)
                label[int(file.split('.')[0])] = 1.0
                self.labels.append(torch.from_numpy(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]
