import os
import torch
import torch.nn as nn
from config import Config
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from dataloader import ImageDataset
import numpy as np

args = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    device = torch.device('cuda')
    model = models.resnet50(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 3)
    model.to(device)

    # model.load_state_dict(torch.load(args.param_file))

    EPOCHS = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        dataset = ImageDataset(args.file_path)
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers,
                                drop_last=args.drop_last)
        for img, label in dataloader:
            img = np.transpose(img, [0, 3, 1, 2])
            img = img.to(device)
            label = label.to(device)
            predict = model(img)

            loss = F.pairwise_distance(predict, label, p=2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {}, Training loss is {}'.format(epoch, loss.item()))
            if min_loss > loss.item():
                min_loss = loss.item()
                torch.save(model.state_dict(), args.param_file)
        print('Epoch {}, min loss is {}'.format(epoch, min_loss))
