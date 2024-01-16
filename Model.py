import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# define customize model 
class ParticleClassifier(nn.Module):
    def __init__(self, input_size): # n_data*16, n_data*7*7
        super(ParticleClassifier, self).__init__()
        self.scalar_layer = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.img_layer = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, padding=1),  # add padding to remain size
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # size, stride
        )

        # size after convolution
        # out of Conv2D = (in + 2 * padding - kernal_size) / stride + 1
        # out of MaxPool2D = (in - kernal_size) / stride + 1
        conv_size = (7 - 2) // 2 + 1

        # fully connected layer
        self.fc = nn.Linear(in_features=16 + conv_size * conv_size, out_features=2)  # 調整全連接層的輸入尺寸和輸出尺寸

        # Softmax for binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_scalar, x_image):
        out1 = self.scalar_layer(x_scalar)
        out2 = self.img_layer(x_image)
        out2 = torch.flatten(out2, start_dim=1)
        out = torch.concat((out1, out2), dim=1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

# if __name__ == '__main__':

