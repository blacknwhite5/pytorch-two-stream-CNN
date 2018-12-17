#!/usr/bin/env python
# -*- conding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from net import SpatialNet, TemporalNet
from torch.utils.data import DataLoader
from torchvision import datasets

import os
import numpy as np
import argparse
from preprocess import Preprocess

# args
parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()

# path
pretrained = 'data/pretrained'
params_spatial = '/spatial.pth'
params_temporal = '/temporal.pth'

# hyper-params
epochs = 100
batch_size = 10
lr = 0.001
momentum = 0.9


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre-processing
Preprocess()

# 신경망 구성
spatialnet = SpatialNet().to(device)
temporalnet = TemporalNet().to(device)

print(spatialnet)
print(temporalnet)


# 신경망 파라매터 로드
if os.path.isfile(pretrained+params_spatial):
    net.load_state_dict(torch.load(pretrained+params_spatial))
    net.load_state_dict(torch.load(pretrained+params_temporal))
    print('\n[*]parameters loaded')


# loss function, optimizer 정의
criterion = nn.CrossEntropyLoss()
optim_rgb = optim.SGD(spatialnet.parameters(), lr=lr, momentum=momentum)
optim_opt = optim.SGD(temporalnet.parameters(), lr=lr, momentum=momentum)
print(optim_rgb)


# 데이터 전처리 정의
transform = transforms.Compose([
         transforms.Resize(255),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])


# 신경망 학습
def train():

    # RGB 이미지 데이터 로더
    rgb_data = datasets.ImageFolder(root='data/image/',
                                    transform=transform)
    
    spatial_loader = DataLoader(dataset=rgb_data,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)

    # optical flow 데이터 로더
    optical_data = datasets.ImageFolder(root='data/optical/',
                                       transform=transform)
    
    temporal_loader = DataLoader(dataset=optical_data,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=True)


    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, ((img, rgb_target), (optical, opt_target)) in enumerate(zip(spatial_loader, temporal_loader)):
            img = img.to(device)
            rgb_target = rgb_target.to(device)

            optical = optical.to(device)
            opt_target = opt_target.to(device)

            optim_rgb.zero_grad()
            optim_opt.zero_grad()

            outputs_spatial = spatialnet(img)
            pred_spatial = torch.max(outputs_spatial, 1)[1]

            outputs_temporal = temporalnet(optical)
            pred_temporal = torch.max(outputs_temporal, 1)[1]
            

            loss_spatial = criterion(outputs_spatial, rgb_target)
            loss_temporal = criterion(outputs_temporal, opt_target)
            loss = loss_spatial + loss_temporal

            loss.backward()
            optim_rgb.step()
            optim_opt.step()

            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print('---Spatial---')
                print('prediction  : {0} \ntarget      : {1}'.format(pred_spatial, rgb_target))
                print('---Temporal--')
                print('prediction  : {0} \ntarget      : {1}'.format(pred_temporal, opt_target))
                print('[epoch : {0:3d}, {1:5d}/{2}] loss : {3:3f}'.format(epoch+1, (i+1)*batch_size, len(optical_data), running_loss/((i+1)*batch_size)))
                print('') 
    print('Training Finished')


    # 파라매터 저장
    if not os.path.exists(pretrained):
        os.makedirs(pretrained, exist_ok=True)
    torch.save(spatialnet.state_dict(), pretrained+params_spatial)
    torch.save(temporalnet.state_dict(), pretrained+params_temporal)

def test():
    rgb_data = datasets.ImageFolder(root='data/image/',
                                    transform=transform)

    spatial_loader = DataLoader(dataset=rgb_data,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=True)
    
    running_loss = 0.0
    correct = 0

    for i, (img, target) in enumerate(dataloader):
        img = img.to(device)
        targets = target.to(device)

        optimizer.zero_grad()

        outputs = net(img)
        prediction = torch.max(outputs, 1)[1]

        loss = criterion(outputs, targets)
        running_loss += loss.item()
        correct += sum(targets.cpu().numpy() == prediction.cpu().numpy())
        print('prediction   : {0} \ntarget       : {1} \nloss         : {2}'.format(prediction, target, running_loss/(i+1)))
    print('accuracy     : {}'.format(100*correct/((i+1)*batch_size)))
    print('Test Finished')




def main():
    if args.test:
        test()
    else:
        train()

if __name__ == '__main__':
    main()
