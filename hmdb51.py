import torch
import torchvision.transforms as transforms

import os
import glob
import cv2
import random
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from pprint import pprint

class HMDB51(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, ratio=0.7):
        self.transform = transform
        self.train = train
        self.data = {}
        self.onehot_label = {}

        video_list = [avi for avi in glob.iglob('**/*.avi', recursive=True)]
        self.labels = sorted(os.listdir(os.path.join(os.getcwd(), 'data/hmdb51_org')))

        # One-hot encoding
        self.onehot_encoding()
        
        # 데이터 전처리 (video -> image)
        if not os.path.exists('data/image'):
            for label in self.labels:
                os.makedirs(os.path.join(os.getcwd(), 'data/image', label), exist_ok=True)

            for video in video_list:
                self.video2frame(video, os.path.join(os.getcwd(), 'data/image'))

        # {이미지 : 레이블}
        image_list = glob.glob('data/image/**/*.jpg', recursive=True)
        for image in image_list:
            self.data[image] = self.onehot_label[image.split('/')[-2]]

   
        # train, test split
        split_idx = int(len(image_list) * ratio)
        # random.seed(3)
        random.shuffle(image_list)
        self.train_image, self.test_image = image_list[:split_idx], image_list[split_idx:]
        
        self.train_label = [self.data[image] for image in self.train_image]
        self.test_label = [self.data[image] for image in self.test_image] 
        

    def __getitem__(self, idx):
        if self.train:
            img, target = self.train_image[idx], self.train_label[idx]

        else:
            img, target = self.test_image[idx], self.test_label[idx]

        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self):
        if self.train:
            return len(self.train_image)
        else:
            return len(self.test_image)



    def onehot_encoding(self):
        for i, label_name in enumerate(self.labels):
            onehot = [0] * len(self.labels)
            onehot[i] = 1
            self.onehot_label[label_name] = np.array(onehot)


    def video2frame(self, video, save_path, count=0):
        folder_name, video_name= video.split('/')[-2], video.split('/')[-1]

        capture = cv2.VideoCapture(video)
        get_frame_rate = round(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 16)

        while True:
            ret, image = capture.read()
            if not ret:
                break
          
            if(int(capture.get(1)) % get_frame_rate == 0):
                count += 1
                fname = '/{0}_{1:05d}.jpg'.format(video_name, count)
                cv2.imwrite('{}/{}/{}'.format(save_path, folder_name, fname), image)

        print("{} images are extracted in {}.". format(count, save_path))




def main():
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                ])

    hmdb51 = HMDB51(transform=transform,
                   train=True)

    dataloader = DataLoader(dataset=hmdb51,
                    batch_size=10,
                    num_workers=2,
                    shuffle=True)

    for i, (img, label) in enumerate(dataloader):
        print(img)
        print(label)



if __name__ == '__main__':
    main()   
