import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


def setup_mnist_net(weights):
    net = Net()
    net.eval()
    net.dropout1 = nn.Dropout2d(p=0)
    net.dropout2 = nn.Dropout2d(p=0)
    net.load_state_dict(torch.load(weights))
    return net


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output

    def run(self, image):
        with torch.no_grad():
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
            image = torch.tensor(image, dtype=torch.float)
            image = Variable(image, requires_grad=False)
            output = self(image)
            prediction = output.data.max(1, keepdim=True)[1]
            return prediction
