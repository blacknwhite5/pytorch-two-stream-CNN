#!/usr/bin/env python
# -*- conding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from hmdb51 import HMDB51
from torch.utils.data import DataLoader

import os
import numpy as np


# hyper-params
epochs = 100
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 51),
            nn.Softmax(dim=1)            
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

# 신경망 구성
net = Net().to(device)
print(net)

# loss function, optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print(optimizer)


# 데이터 전처리
transform = transforms.Compose([
         transforms.Resize(255),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])


# 신경망 학습
def train():
    hmdb51 = HMDB51(transform=transform,
                    train=True)

    dataloader = DataLoader(dataset=hmdb51,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (img, target) in enumerate(dataloader):
            img = img.to(device)
            targets = target.to(device)

            optimizer.zero_grad()

            outputs = net(img)
            prediction = torch.max(outputs, 1)[1]
            loss = criterion(outputs, torch.max(targets, 1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print('[epoch : {0:3d}, {1:5d}/{2}] loss : {3:3f}'.format(epoch+1, i+1, len(hmdb51), running_loss/(i+1)))
    
    print('Training Finished')



def main():
    train()

if __name__ == '__main__':
    main()
