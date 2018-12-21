import os
import glob
import cv2
import random
import numpy as np


class Preprocess():
    def __init__(self, ratio=.7):
        self.ratio = ratio
        
        labels = sorted(os.listdir(os.path.join(os.getcwd(), 'data/hmdb51_org')))

        video_list = []
        for label in labels:
            video_list.append([avi for avi in glob.iglob('data/hmdb51_org/{}/*.avi'.format(label), recursive=True)])

        # 레이블 인덱싱
        label_index = {label : np.array(i) for i, label in enumerate(labels)}
            
        # 데이터 전처리 (video -> image)
        if not os.path.exists('data/train'):
            for label in labels:
                os.makedirs(os.path.join(os.getcwd(), 'data/train/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/test/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/train/optical', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/test/optical', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/val/image', label), exist_ok=True)
                os.makedirs(os.path.join(os.getcwd(), 'data/val/optical', label), exist_ok=True)
    

            for videos in video_list:
                for i, video in enumerate(videos):
                    # train
                    if i < round(len(videos)*self.ratio):
                        self.video2frame(video, 'data/train/image', 'data/train/optical')

                    # validation
                    elif i > round(len(videos)*0.9):
                        self.video2frame(video, 'data/val/image/', 'data/val/optical/')

                    # test
                    else:
                        self.video2frame(video, 'data/test/image', 'data/test/optical')


    
    def video2frame(self, video, frame_save_path,optical_save_path, count=0):
        '''
            1개의 동영상 파일에서 약 16 프레임씩 이미지(.jpg)로 저장
    
            args
                video : 비디오 파일 이름
                save_path : 저장 경로
    
        '''
        folder_name, video_name= video.split('/')[-2], video.split('/')[-1]

        capture = cv2.VideoCapture(video)
        get_frame_rate = round(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 16)

        _, frame = capture.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        hsv = np.zeros_like(frame) # Farneback 알고리즘 이용하기 위한 초기화
        hsv[..., 1] = 255 # 초록색 바탕 설정

        while True:
            ret, image = capture.read()
            if not ret:
                break

            if(int(capture.get(1)) % get_frame_rate == 0):
                count += 1
                fname = '/{0}_{1:05d}.jpg'.format(video_name, count)
                cv2.imwrite('{}/{}/{}'.format(frame_save_path, folder_name, fname), image)

                next_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                fname = '/{0}_{1:05d}_flow.jpg'.format(video_name, count)
                cv2.imwrite('{}/{}/{}'.format(optical_save_path, folder_name, fname), rgb)

            prvs = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)


        print("{} images are extracted in {}.". format(count, frame_save_path))
