# +
import os
import sys
sys.path.append("../")

import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast
from utils.io import *
# -

import numpy as np


class Bios:
    def __init__(self, path, step_size=5):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased")
        self.train, self.test = self.load(os.path.join(
            path, "train.pkl")), self.load(os.path.join(path, "test.pkl"))
        self.num_classes = max(self.train['y']) + 1

        sizes = []
        for _ in range(self.num_classes):
            sizes.append(len(self.train['y'][self.train['y'] == _]))

        self.order = np.argsort(sizes)[::-1]
        self.o2id = {k: v for v, k in enumerate(self.order)}
        self.step_size = step_size
        self.iterations = len(self.order) // self.step_size

    def load(self, filename):
        content = []
        with open(filename, "rb") as f:
            content = pickle.load(f)

        data, y_label, g_label = [], [], []
        for d, txt, y, g in tqdm(content):

            text_emb = [self.tokenizer.encode(txt, \
                                                  max_length = 32, \
                                                  pad_to_max_length =True, \
                                                  truncation=True, \
                                                  add_special_tokens=True)]
            data.append(text_emb[0])
            y_label.append(y)
            g_label.append(g)

        dataset = {}
        dataset['data'] = np.array(data)
        dataset['y'] = np.array(y_label)
        dataset['g'] = np.array(g_label)
        return dataset

    def get_next_group(self, step):
        # step cannot be greater than number of classes
        if (step) * self.step_size >= max(self.train['y']) + 1:
            raise

        # take 2 classes at the start
        labels = []
        #         if step == 0:
        #             labels.append(0)
        #         labels.append(step + 1)

        label_upp_limit = min(
            max(self.train['y']) + 1, (step + 1) * self.step_size)
        labels.extend(list(range(step * self.step_size, label_upp_limit)))
        idx = self.get_indices(self.train, labels)

        train_group = self.get_data(self.train, idx)

        new_test_idx = self.get_indices(self.test, labels)
        new_test_group = self.get_data(self.test, new_test_idx)

        old_test_idx = [] if step == 0 else self.get_indices(
            self.test, list(range(label_upp_limit)))
        old_test_group = self.get_data(self.test, old_test_idx)

        return train_group, new_test_group, old_test_group
    
    def get_cumulative_group(self, step):
        # step cannot be greater than number of classes
        if step * self.step_size > max(self.train['y']) + 1:
            raise Exception('step size too high!')
            
        if step == 0:
            raise Exception('No old training data for step 0!')
            
        
        labels = []
        
        label_upp_limit = min(
            max(self.train['y']) + 1, (step + 1) * self.step_size)
        labels.extend(list(range(label_upp_limit)))
        
        idx = self.get_indices(self.train, labels)
        old_train_group = self.get_data(self.train, idx)
        
        return old_train_group
        

    def get_data(self, arr, idx):
        y_labels = [self.o2id[y] for y in arr['y'][idx]]
        return (arr['data'][idx], y_labels, arr['g'][idx])

    def get_indices(self, arr, labels):
        idx = []
        for l in labels:
            idx.extend(np.where(arr['y'] == self.order[l])[0])
        return idx


class BatchData(Dataset):
    def __init__(self, text_emb, y_labels, g_labels):
        self.text_emb = text_emb
        self.y_labels = y_labels
        self.g_labels = g_labels

    def __getitem__(self, index):
        text = torch.FloatTensor(self.text_emb[index])
        y_label = self.y_labels[index]
        g_label = self.g_labels[index]

        return text, y_label, g_label

    def __len__(self):
        return len(self.text_emb)


def unit_test():
    def iterate(bios):
        for i in range(bios.iterations + 1):
            train_group, new_test_group, old_test_group = bios.get_next_group(
                i)
            g = train_group[2]
            split = len(np.where(g == 0)[0]) / len(g)
            print(f"{i} - {split}")

    bios = Bios("../../../purging-embeddings/data/bios")
    iterate(bios)
