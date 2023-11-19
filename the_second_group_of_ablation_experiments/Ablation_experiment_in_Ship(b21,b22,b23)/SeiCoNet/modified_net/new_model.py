import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # Convolutional layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    # Forward propagation
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # Model weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



