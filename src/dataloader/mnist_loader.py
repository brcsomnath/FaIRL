# +
import os
import sys

sys.path.append("../")

import torch
from utils.io import *

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision import transforms
# -

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
])


class ColoredMNIST(VisionDataset):
    def __init__(self,
                 root='./data',
                 env='train',
                 transform=None,
                 target_transform=None):
        super(ColoredMNIST, self).__init__(root,
                                           transform=transform,
                                           target_transform=target_transform)
        self.data_label_tuples = torch.load(
            os.path.join(self.root, f'{env}.pt'))

    def __getitem__(self, index):
        img, target, protected = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, protected

    def __len__(self):
        return len(self.data_label_tuples)


class BiasedMNIST:
    def __init__(self,
                 dataset_path='../../data/colored_mnist',
                 step_size=2,
                 num_target_class=10):
        self.train_set = ColoredMNIST(root=dataset_path,
                                      env='train',
                                      transform=img_transform)
        self.test_set = ColoredMNIST(root=dataset_path,
                                     env='test',
                                     transform=img_transform)

        self.num_target_class = num_target_class
        self.step_size = step_size
        self.iterations = self.num_target_class // self.step_size

    def get_next_group(self, step):
        # step cannot be greater than number of classes
        if (step) * self.step_size >= self.num_target_class + 1:
            raise

        labels = []

        label_upp_limit = min(self.num_target_class,
                              (step + 1) * self.step_size)
        labels.extend(list(range(step * self.step_size, label_upp_limit)))
        idx = self.get_indices(self.train_set, labels)

        train_group = self.get_data(self.train_set, idx)

        new_test_idx = self.get_indices(self.test_set, labels)
        new_test_group = self.get_data(self.test_set, new_test_idx)

        old_test_idx = [] if step == 0 else self.get_indices(
            self.test_set, list(range(label_upp_limit)))
        old_test_group = self.get_data(self.test_set, old_test_idx)

        return train_group, new_test_group, old_test_group

    def get_cumulative_group(self, step):
        # step cannot be greater than number of classes
        if (step + 1) * self.step_size > self.num_target_class:
            raise Exception('step size too high!')

        if step == 0:
            raise Exception('No old training data for step 0!')

        labels = []
        label_upp_limit = min(self.num_target_class,
                              (step + 1) * self.step_size)
        labels.extend(list(range(label_upp_limit)))

        idx = self.get_indices(self.train_set, labels)
        old_train_group = self.get_data(self.train_set, idx)

        return old_train_group

    def get_indices(self, arr, labels):
        idx = []
        target_labels = np.array([_[1] for _ in arr])

        for l in labels:
            idx.extend(np.where(target_labels == l)[0])
        return idx

    def get_data(self, arr, idx):
        return [arr[i] for i in idx]


def unit_test():
    def iterate(cmnist):
        for i in range(cmnist.iterations):
            train_group, new_test_group, old_test_group = cmnist.get_next_group(
                i)
            g = np.array([grp[2] for grp in train_group])
            split = len(np.where(g == 0)[0]) / len(g)
            print(f"{i} - {split}")

    mnist = BiasedMNIST()
    iterate(mnist)