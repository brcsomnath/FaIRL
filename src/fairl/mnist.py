import argparse
import os
import sys

sys.path.append("../")
sys.path.append("src/")

import warnings
from copy import deepcopy
from random import *

warnings.filterwarnings("ignore")

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from apricot import (CustomSelection, FacilityLocationSelection,
                     FeatureBasedSelection)
from dataloader.mnist_loader import BiasedMNIST
from fairl.loss import MCR
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from tensorboard_logger import configure, log_value
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, tqdm_notebook
from transformers import BertModel, BertTokenizerFast, DistilBertModel
from utils.evaluate import *
from utils.io import *
from utils.net import ConvNet, Net


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        default=".95",
                        type=str,
                        help="Dataset name.")
    parser.add_argument("--dataset_path",
                        default="../../data/colored_mnist",
                        type=str,
                        help="Dataset path.")
    parser.add_argument(
        "--MODEL",
        default="",
        type=str,
        help="",
    )
    parser.add_argument("--step_size",
                        default=2,
                        type=int,
                        help="Number of classes in a training step.")
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="Batch Size.")
    parser.add_argument("--num_layers",
                        default=3,
                        type=int,
                        help="Number of layers.")
    parser.add_argument("--device",
                        default="cuda:3",
                        type=str,
                        help="GPU device.")
    parser.add_argument(
        "--model_save_path",
        default="saved_models/",
        type=str,
        help="Save path of the models.",
    )
    parser.add_argument(
        "--embedding_size",
        default=1024,
        type=int,
        help=
        "Hidden size of the representation output of the feature extractor.",
    )
    parser.add_argument("--epochs",
                        default=2,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("--DATA_IDX", default=0, type=int, help="Data ID")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--mom",
                        type=float,
                        default=0.9,
                        help="momentum (default: 0.9)")
    parser.add_argument("--wd",
                        type=float,
                        default=5e-4,
                        help="weight decay (default: 5e-4)")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="beta",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.,
        help="gamma",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="eta",
    )
    parser.add_argument(
        "--gam1",
        type=float,
        default=0.0,
        help="gamma1 for tuning empirical loss (default: 1.)",
    )
    parser.add_argument(
        "--gam2",
        type=float,
        default=0.0,
        help="gamma2 for tuning empirical loss (default: 1.)",
    )
    parser.add_argument(
        "--gam3",
        type=float,
        default=1.0,
        help="gamma3 for tuning empirical loss (default: 1.)",
    )
    parser.add_argument("--eps",
                        type=float,
                        default=0.5,
                        help="eps squared (default: 0.5)")

    parser.add_argument("--num_target_class", default=10, type=int)
    parser.add_argument("--num_protected_class", default=10, type=int)

    parser.add_argument("--exemplar_selection", default='random', 
    choices=['random', 'prototype', 'submod'], type=str)
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--num_components", default=10, type=int)
    return parser


class FaIRL:
    def __init__(self,
                 dataset,
                 embedding_size=1024,
                 device='cuda:0',
                 save_dir='../../model/cmnist/',
                 num_layers=2):
        self.dataset = dataset
        self.embedding_size = embedding_size
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.step_size = self.dataset.step_size
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.num_layers = num_layers

    def save_models(self, netD, netG, conv_net, step):
        torch.save(netD, os.path.join(self.save_dir, f'netD-{step}.pb'))
        torch.save(netG, os.path.join(self.save_dir, f'netG-{step}.pb'))
        torch.save(conv_net,
                   os.path.join(self.save_dir, f'bert-{step}.pb'))

    def generate_debiased_embeddings(self, dataset_loader, conv_net, netG):
        """
        Retrieve debiased embeddings post training 

        Arguments:
            args: arguments
            dataset_loader: pytorch data loader
            net: \phi(x) network

        Return:
            dataset: [debiased_embedding, y, z]
        """

        dataset = []
        for data, y, z in (dataset_loader):
            real_data = data.to(self.device)

            with torch.no_grad():
                output = netG(conv_net(real_data))

            purged_emb = output.detach().cpu().numpy()
            data_slice = [(data, int(y.detach().cpu().numpy()),
                           int(z.detach().cpu().numpy()))
                          for data, y, z in zip(purged_emb, y, z)]
            dataset.extend(data_slice)
        return dataset

    def sample_exemplars(self, args, labels, data_train):
        reserve_idx = []
        for l in range(min(labels), max(labels) + 1):
            label_set = [i for i, _ in enumerate(labels) if _ == l]
            if args.exemplar_selection == 'random':
                # Random sampling
                reserve_idx.extend(sample(label_set, args.num_samples))
            elif args.exemplar_selection == 'submod':
                # submodular optimization
                feature_set = np.array([data_train[i] for i in label_set])
                submod = FacilityLocationSelection(args.num_samples,
                                                   metric='euclidean',
                                                   optimizer='lazy',
                                                   random_state=0,
                                                   verbose=False)
                submod.fit(feature_set)
                indices = [label_set[r] for r in submod.ranking]
                reserve_idx.extend(indices)
            elif args.exemplar_selection == 'prototype':
                feature_set = np.array([data_train[i] for i in label_set])

                pca = PCA(n_components=args.num_components)
                pca.fit(feature_set)

                eigen_vecs = pca.components_
                top_args = np.argsort(np.dot(eigen_vecs, feature_set.T),
                                      axis=1)

                indices = []
                for vec in top_args:
                    count = 0
                    for v in vec:
                        if v not in indices:
                            indices.append(v)
                            count += 1

                        if count == args.num_samples // args.num_components:
                            break
                indices = [label_set[r] for r in indices]
                reserve_idx.extend(indices)
        return reserve_idx

    def trainer(self, args):

        reserve_new = []

        previous_conv = None
        previous_netG = None

        new_acc = []
        new_leakages = []
        new_dps = []
        new_tprs = []

        old_acc = []
        old_leakages = []
        old_dps = []
        old_tprs = []

        iteration = 0

        for step in range(self.dataset.iterations):
            train, test, old_test = self.dataset.get_next_group(step)

            train_data = DataLoader(train,
                                    batch_size=args.batch_size,
                                    shuffle=True)
            val_data = DataLoader(test,
                                  batch_size=args.batch_size,
                                  shuffle=False)
            old_val_data = DataLoader(old_test,
                                      batch_size=args.batch_size,
                                      shuffle=False)

            if step == 0:
                conv_net = ConvNet(args.embedding_size)
                conv_net.to(self.device)

                netG = Net(args.embedding_size, args.num_layers)
                netG.to(self.device)

            elif step > 0:
                previous_conv = conv_net
                previous_netG = netG

            netD = Net(args.embedding_size, args.num_layers)
            netD.to(self.device)

            mcr_loss = MCR(gam1=args.gam1,
                           gam2=args.gam2,
                           gam3=args.gam3,
                           eps=args.eps,
                           numclasses=args.num_target_class)

            optim_G = optim.Adam([{
                "params": conv_net.parameters()
            }, {
                "params": netG.parameters()
            }],
                                 lr=2e-5,
                                 betas=(0.5, 0.999))

            optim_D = optim.Adam([{
                "params": netD.parameters()
            }],
                                 lr=2e-5,
                                 betas=(0.5, 0.999))

            LN = nn.LayerNorm(args.embedding_size, elementwise_affine=False)

            itr = tqdm(range(args.epochs))

            for i, epoch in enumerate(itr, 0):
                total_task_loss = 0.
                total_bias_loss = 0.
                total_recon_loss = 0.
                total_loss = 0.

                conv_net.train()
                netD.train()
                netG.train()

                if step > 0:
                    old_batch, old_label, old_bias = next(iter(reserve))
                    old_data = old_batch.to(self.device)

                    Z_old = previous_netG(previous_conv(old_data)).detach()

                    total_labels = max(old_label) + 1
                    _, (_, _, z_old_losses,
                        _) = mcr_loss.deltaR(LN(Z_old), old_label,
                                             total_labels)

                for _, (batch, label, bias) in enumerate(train_data):
                    real_data = batch.to(self.device)

                    conv_net.zero_grad()
                    netG.zero_grad()
                    netD.zero_grad()

                    # update discriminator first
                    optim_D.zero_grad()
                    optim_G.zero_grad()

                    Z = netG(conv_net(real_data))

                    Z_bar = netD(Z.detach())

                    disc_loss, comp = mcr_loss.deltaR(LN(Z_bar), bias,
                                                      args.num_protected_class)
                    disc_loss.backward()
                    optim_D.step()

                    # update generator
                    conv_net.zero_grad()
                    netG.zero_grad()
                    netD.zero_grad()

                    optim_D.zero_grad()
                    optim_G.zero_grad()

                    Z = netG(conv_net(real_data))
                    Z_bar = netD(Z)

                    task_loss, _ = mcr_loss.deltaR(
                        LN(Z), label,
                        min((step + 1) * self.step_size,
                            args.num_target_class))
                    bias_loss, _ = mcr_loss.deltaR(LN(Z_bar), bias,
                                                   args.num_protected_class)

                    loss = task_loss - args.beta * bias_loss

                    old_loss = 0.
                    old_bias_loss = 0.
                    if step > 0:
                        Z = netG(conv_net(old_data))

                        _, (R_z, _, z_losses,
                            _) = mcr_loss.deltaR(LN(Z), old_label,
                                                 total_labels)

                        iteration += 1

                        R_zjzjold = 0.
                        for j in range(total_labels):
                            new_z = torch.cat(
                                (Z[old_label == j], Z_old[old_label == j]), 0)
                            R_zjzjold += mcr_loss.compute_discrimn_loss(
                                LN(new_z).T)

                        old_loss = (R_zjzjold - 0.25 * sum(z_losses) -
                                    0.25 * sum(z_old_losses))

                        loss += args.gamma * old_loss

                        Z_bar = netD(Z)
                        old_bias_loss, _ = mcr_loss.deltaR(
                            LN(Z_bar), old_bias, args.num_protected_class)
                        loss = loss - args.eta * old_bias_loss

                    loss.backward()
                    optim_G.step()

                    itr.set_description(f"Loss = {-(loss.item()):.6f} \
                                        Task Loss = {-(task_loss.item()):.6f} \
                                        Bias Loss = {(-disc_loss.item()):.6f} \
                                        Old Bias Loss = {(0. if old_bias_loss == 0. else -old_bias_loss.item()):.6f} \
                                        Recon Loss = {(0. if old_loss == 0. else -old_loss.item()):.6f}"
                                        )
            print("training complete.")

            print("====================================================")
            print(f" Step - {step}")

            print("Generating representations ...")

            train_data = DataLoader(train,
                                    batch_size=args.batch_size,
                                    shuffle=False)

            train_dataset = self.generate_debiased_embeddings(
                train_data, conv_net, netG)
            test_dataset = self.generate_debiased_embeddings(
                val_data, conv_net, netG)
            old_test_dataset = self.generate_debiased_embeddings(
                old_val_data, conv_net, netG)
            print("Done!")

            data_train, task_label_train, bias_label_train = get_data(
                train_dataset)
            data_test, task_label_test, bias_label_test = get_data(
                test_dataset)
            old_data_test, old_task_label_test, old_bias_label_test = get_data(
                old_test_dataset)
            print("Sampling exemplars ...")

            labels = [x[1] for x in train]
            reserve_idx = self.sample_exemplars(args, labels, data_train)
            reserve_new.extend([train[idx] for idx in reserve_idx])

            reserve = DataLoader(reserve_new,
                                 batch_size=len(reserve_new),
                                 shuffle=True)
            print(f"Exemplar length: {len(reserve)}")
            print("Done!")

            self.save_models(netD, netG, conv_net, step)

            F1, y_pred_task = evaluate_mlp(data_train, task_label_train,
                                           data_test, task_label_test)
            print(f"Task: F1 - {F1}")
            new_acc.append(F1)

            bias_F1, y_pred_bias = evaluate_mlp(data_train, bias_label_train,
                                                data_test, bias_label_test)
            print(f"Bias: F1 - {bias_F1}")

            maj_baseline = len(
                bias_label_test[bias_label_test == 0]) / len(bias_label_test)
            leakage = abs(
                max(100 - bias_F1, bias_F1) -
                100 * max(maj_baseline, 1 - maj_baseline))
            new_leakages.append(leakage)
            print(f"Majority Baseline: {maj_baseline} Leakage: {leakage}")

            dp = get_demographic_parity(y_pred_task, bias_label_test)
            tpr = TPR_rms(task_label_test, y_pred_task, bias_label_test)
            print(f"DP: {dp} TPR: {tpr}")

            new_dps.append(dp)
            new_tprs.append(tpr)

            if step == 0:
                old_acc.append(F1)
                old_leakages.append(leakage)
                old_dps.append(dp)
                old_tprs.append(tpr)

            if step > 0:
                old_train = self.dataset.get_cumulative_group(step)
                old_train_data = DataLoader(old_train,
                                            batch_size=args.batch_size,
                                            shuffle=True)
                old_train_dataset = self.generate_debiased_embeddings(
                    old_train_data, conv_net, netG)

                old_data_train, old_task_label_train, old_bias_label_train = get_data(
                    old_train_dataset)

                F1, y_pred_task_old = evaluate_mlp(old_data_train,
                                                   old_task_label_train,
                                                   old_data_test,
                                                   old_task_label_test)
                print(f"Old Task: F1 - {F1}")
                old_acc.append(F1)

                F1, y_pred_bias_old = evaluate_mlp(old_data_train,
                                                   old_bias_label_train,
                                                   old_data_test,
                                                   old_bias_label_test)
                print(f"Old Bias: F1 - {F1}")

                maj_baseline = len(old_bias_label_test[
                    old_bias_label_test == 0]) / len(old_bias_label_test)
                old_leakage = abs(
                    max(100 - F1, F1) -
                    100 * max(maj_baseline, 1 - maj_baseline))
                old_leakages.append(old_leakage)

                old_dp = get_demographic_parity(y_pred_task_old,
                                                old_bias_label_test)
                old_tpr = TPR_rms(old_task_label_test, y_pred_task_old,
                                  old_bias_label_test)
                print(
                    f"Majority baseline: {maj_baseline} Leakage: {old_leakage}"
                )
                print(f"Old DP: {old_dp}  TPR: {old_tpr}")

                old_dps.append(old_dp)
                old_tprs.append(old_tpr)

            print("====================================================")

        print(f"Avg. new task accuracy: {np.array(new_acc).mean()}")
        print(f"Avg. new task leakage: {np.array(new_leakages).mean()}")
        print(f"Avg. new task DP: {np.array(new_dps).mean()}")
        print(f"Avg. new task TPR: {np.array(new_tprs).mean()}")

        print(f"Avg. old task accuracy: {np.array(old_acc).mean()}")
        print(f"Avg. old task leakage: {np.array(old_leakages).mean()}")
        print(f"Avg. old task DP: {np.array(old_dps).mean()}")
        print(f"Avg. old task TPR: {np.array(old_tprs).mean()}")


if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mnist = BiasedMNIST(
        dataset_path=f'{args.dataset_path}-{args.dataset}', step_size=args.step_size)

    trainer = FaIRL(mnist, device=args.device)

    trainer.trainer(args)
