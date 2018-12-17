import torch.nn as nn

class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            # nn.batchnorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            # nn.batchnorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            # nn.batchnorm2d(512),            
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 5),
            
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x





class TemporalNet(nn.Module):
    def __init__(self):
        super(TemporalNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            # nn.batchnorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            # nn.batchnorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3),
            # nn.batchnorm2d(512),            
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 5),
            
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x



