import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class leanCNN(nn.Module):
    """An implementation of a lean CNN with just 2 convolutional layers and 3 fully-connected layers.
    Args:
       num_class: Number of class for classification.
    """

    def __init__(self, num_class, input_fdim, input_tdim):
        super().__init__()
        self.fdim = input_fdim
        self.tdim = input_tdim
        self.conv1 = nn.Conv2d(1, 2, 3)  # (6 ,30, 30)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 15, 15)
        self.conv2 = nn.Conv2d(2, 64, 3)  # (16, 13, 13)
        self.neurons = self.calc_flatten_fc_neuron()
        self.fc1 = nn.Linear(
            self.neurons, 256
        )  # (oiginal with mfcc - 2400, 128) (12000, 128 for melspec)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def conv_block(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def calc_flatten_fc_neuron(self):
        size = self.conv_block(torch.randn(1, 1, self.fdim, self.tdim)).size()
        m = 1
        for i in size:
            m *= i
        return int(m)

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    batch_size = 32
    input_fdim = 32
    input_tdim = 93
    num_classes = 7

    cnn_mdl = leanCNN(num_class=num_classes, input_fdim=input_fdim, input_tdim=input_tdim)
    test_input = torch.rand([batch_size, 1, input_fdim, input_tdim])
    test_output = cnn_mdl(test_input)
    print(test_output.shape)

    ## print(summary(cnn_mdl, input_size=(batch_size, 1, input_fdim, input_tdim)))