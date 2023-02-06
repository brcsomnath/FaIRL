from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, embedding_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(embedding_size, embedding_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(embedding_size, embedding_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, embedding_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        feature = self.fc1(x)
        return (feature)


class Classifier(nn.Module):
    def __init__(self, embedding_size):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        feature = self.fc1(x)
        return self.fc2(feature)


class FFNN(nn.Module):
    def __init__(self, embedding_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 3, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.fc4 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28 * 28 * 3)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        feature = self.fc4(x)
        return feature
