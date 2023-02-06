# +
import os
import sys

sys.path.append("../")
import torch
import argparse
import matplotlib.pyplot as plt

# +
from tqdm import tqdm
from dataloader.bios import BatchData
from torch.utils.data import DataLoader
from utils.clf import train_mlp
from utils.io import *
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from collections import defaultdict, Counter
from transformers import BertModel
from tqdm import tqdm, tqdm_notebook
from random import sample

import matplotlib.cm as cm
# -

import umap


def load_bert(filename):
    bert_model = torch.load(filename)
    return bert_model


def generate_debiased_embeddings(dataset_loader, model, netG, device):
    dataset = []
    for data, y, z in tqdm(dataset_loader):
        real_data = data.to(device).long()

        with torch.no_grad():
            bert_output = model(real_data)[1]
            output = netG(bert_output)

        purged_emb = output.detach().cpu().numpy()
        data_slice = [(data, int(y.detach().cpu().numpy()), int(z.detach().cpu().numpy()))
                      for data, y, z in zip(purged_emb, y, z)]
        dataset.extend(data_slice)
    return dataset


def get_data(dataset):
    data = np.array([x[0] for x in dataset])
    y = np.array([x[1] for x in dataset])
    g = np.array([x[2] for x in dataset])
    return data, y, g


def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


bios = load("../../data/bios.pkl")

# +
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = torch.load("../../model/ablations/bert-5.pb")
bert_model.to("cuda:2")

netG = torch.load("../../model/ablations/netG-5.pb")
netG.to("cuda:2")

# +
train, test, old_test = bios.get_next_group(5)

old_val_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
                          batch_size=128,
                          shuffle=False)

old_val_dataset = generate_debiased_embeddings(old_val_data, bert_model, netG, "cuda:2")
data_old_val, y_old_val, g_old_val = get_data(old_val_dataset)
# -

idx = sample(list(range(len(y_old_val))), 20000)
data_old_val_sample = [data_old_val[i] for i in idx]
y_old_val_sample = [y_old_val[i] for i in idx]
g_old_val_sample = [g_old_val[i] for i in idx]

fit = umap.UMAP()
# %time u = fit.fit_transform(data_old_val_sample)

# +
label = g_old_val_sample
colors = cm.rainbow(np.linspace(0, 1, len(set(label))))
colors = ['red', 'black']
markers = ['*', '.']

plt.figure(figsize=(4, 4), dpi=800)
for j in range(len(set(label))):
    x = [_[0] for i, _ in enumerate(u[:10000]) if label[i] == j]
    y = [_[1] for i, _ in enumerate(u[:10000]) if label[i] == j]
    s = [0.1 for _ in range(len(x))]
    plt.scatter(x, y, color=colors[j], marker=markers[j], s=s)

# plt.axis('off') 
plt.xticks([])
plt.yticks([])
plt.show()
# -

# ## Untrained

new_bert_model = BertModel.from_pretrained("bert-base-uncased")
new_bert_model.to("cuda:2")


def get_representations(dataset_loader, model, device):
    dataset = []
    for data, y, z in tqdm(dataset_loader):
        real_data = data.to(device).long()

        with torch.no_grad():
            bert_output = model(real_data)[1]
            output = (bert_output)

        purged_emb = output.detach().cpu().numpy()
        data_slice = [(data, int(y.detach().cpu().numpy()), int(z.detach().cpu().numpy()))
                      for data, y, z in zip(purged_emb, y, z)]
        dataset.extend(data_slice)
    return dataset


old_val_dataset = get_representations(old_val_data, new_bert_model, "cuda:2")
data_old_val, y_old_val, g_old_val = get_data(old_val_dataset)

idx = sample(list(range(len(y_old_val))), 20000)
data_old_val_sample = [data_old_val[i] for i in idx]
y_old_val_sample = [y_old_val[i] for i in idx]
g_old_val_sample = [g_old_val[i] for i in idx]

fit = umap.UMAP()
# %time v = fit.fit_transform(data_old_val_sample)

# +
label = g_old_val_sample
colors = cm.rainbow(np.linspace(0, 1, len(set(label))))
colors = ['red', 'black']
markers = ['*', '.']

plt.figure(figsize=(4, 4), dpi=800)
for j in range(len(set(label))):
    x = [_[0] for i, _ in enumerate(v[:10000]) if label[i] == j]
    y = [_[1] for i, _ in enumerate(v[:10000]) if label[i] == j]
    s = [.1 for _ in range(len(x))]
    plt.scatter(x, y, color=colors[j], marker=markers[j], s=s)

# plt.axis('off') 
plt.xticks([])
plt.yticks([])
plt.show()
