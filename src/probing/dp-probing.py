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

def load_bert(filename):
    bert_model = torch.load(filename)
    return bert_model


def generate_debiased_embeddings(dataset_loader, model, netG, device):
    dataset = []
    for data, y, z in (dataset_loader):
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


def get_demographic_parity(y_hat_main, y_protected):
    """
    Computes Demgraphic parity (DP)

    Arguments:
        y_hat_main: predictions for main task 
        y_protected: protected task labels

    Returns:
        dp: Demographic parity across all labels
    """

    all_y = list(Counter(y_hat_main).keys())

    dp = 0
    for y in all_y:
        D_i = []
        for i in range(2):
            used_vals = y_protected == i
            y_hat_label = y_hat_main[used_vals]
            Di_ = len(y_hat_label[y_hat_label == y]) / len(y_hat_label)
            D_i.append(Di_)
        dp += abs(D_i[0] - D_i[1])

    return dp


def get_TPR(y_main, y_hat_main, y_protected):
    """
    Computes the true positive rate (TPR)

    Arguments:
        y_main: main task labels
        y_hat_main: predictions for main task 
        y_protected: protected task labels

    Returns:
        diffs: different between TPRs
    """

    all_y = list(Counter(y_main).keys())
    y_main, y_hat_main, y_protected = np.array(y_main), np.array(y_hat_main), np.array(y_protected)

    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals["y:{}".format(label)]["p:{}".format(i)] = (y_label == y_hat_label).mean()

    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs


def TPR_rms(y_main, y_hat_main, y_protected):
    """
    Computes the RMS value for a sequence of numbers
    """
    
    _, diffs = get_TPR(y_main, y_hat_main, y_protected)
    return np.sqrt(np.mean(np.square(list(diffs.values()))))


bios = load("../../data/bios.pkl")


def evaluate_mlp(x_train, y_train):
    """
    Evaluates MLPScore for prediction on a task
    """

    clf = MLPClassifier(max_iter=5, verbose=False)

    clf.fit(x_train, y_train)
    return clf


def test_mlp(clf, x_test, y_test):
    """
    Evaluates MLPScore for prediction on a task
    """

    y_pred = clf.predict(x_test)

    F1 = f1_score(y_pred, y_test, average="micro") * 100
    P = precision_score(y_pred, y_test, average="micro") * 100
    R = recall_score(y_pred, y_test, average="micro") * 100
    return F1, y_pred


for step in range(bios.iterations, 0, -1):
    print(f"Step: {step}")

    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = torch.load(f"../../model/ablations/bert-{step}.pb")
    bert_model.to("cuda:2")

    netG = torch.load(f"../../model/ablations/netG-{step}.pb")
    netG.to("cuda:2")
    
    old_train = bios.get_cumulative_group(step)


    old_train_data = DataLoader(BatchData(old_train[0], old_train[1], old_train[2]),
                                    batch_size=128,
                                    shuffle=True)

    old_train_dataset = generate_debiased_embeddings(old_train_data, bert_model, netG, "cuda:2")

    old_data_train, old_y_train, old_g_train = get_data(old_train_dataset)
    clf_final = evaluate_mlp(old_data_train, old_y_train)
    
    for i in range(step+1):
        _, test, old_test = bios.get_next_group(i)

        val_data = DataLoader(BatchData(test[0], test[1], test[2]),
                                  batch_size=128,
                                  shuffle=False)

        val_dataset = generate_debiased_embeddings(val_data, bert_model, netG, "cuda:2")
        data_val, y_val, g_val = get_data(val_dataset)

        F1, y_pred = test_mlp(clf_final, data_val, y_val)

        dp = get_demographic_parity(y_pred, g_val)
        tpr = TPR_rms(y_val, y_pred, g_val)

        print(f"F1: {F1} DP: {dp} TPR: {tpr}")


