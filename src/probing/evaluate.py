import os
import sys

sys.path.append("../")
import torch
import argparse

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


# -

def load_bert(filename):
    bert_model = torch.load(filename)
    return bert_model


def get_representations(bert_model, data_loader, device):
    dataset = []
    for (batch, y, g) in tqdm(data_loader):
        with torch.no_grad():
            # bert_output = netG(bert_model(batch.to(device).long())[1])
            bert_output = bert_model(batch.to(device).long())['last_hidden_state'][:, 0, :].view(-1, 768)

        purged_emb = bert_output.detach().cpu().numpy()
        data_slice = [(data, int(y.detach().cpu().numpy()), int(z.detach().cpu().numpy())) for data, y, z in zip(purged_emb, y, g)]
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


def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels


def nearsub(args, train_features, train_labels, test_features, test_labels):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    num_classes = max(train_labels) + 1  # should be correct most of the time
    features_sort, _ = sort_dataset(train_features,
                                      train_labels,
                                      num_classes=num_classes,
                                      stack=False)
    fd = features_sort[0].shape[1]
    for j in tqdm(range(num_classes)):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j])
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ test_features.T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)

        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = compute_accuracy(test_predict_pca, test_labels)
    acc_svd = compute_accuracy(test_predict_svd, test_labels)
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd


def evaluate_mlp(x_train, y_train, x_test, y_test):
    """
    Evaluates MLPScore for prediction on a task
    """

    clf = MLPClassifier(max_iter=5, verbose=False)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    F1 = f1_score(y_pred, y_test, average="micro") * 100
    return F1, y_pred


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


def probe(args, dataset, stage, device, batch_size=32):
    print(f"Starting stage {stage} ...")
    print("Loading model ...", end="\r")
    bert_model = torch.load(os.path.join(args.model_dir, f"bert-{stage}.pb"))
    bert_model.to(args.device)

    # netG = torch.load(os.path.join(args.model_dir, f"netG-{stage}.pb"))
    # netG.to(device)
    print("Model loaded!")
    
    if stage == 0:
        train, test, old_test = dataset.get_next_group(stage)

        train_data = DataLoader(BatchData(train[0], train[1], train[2]),
                                batch_size=batch_size,
                                shuffle=True)
        val_data = DataLoader(BatchData(test[0], test[1], test[2]),
                              batch_size=batch_size,
                              shuffle=False)
        old_val_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
                                  batch_size=batch_size,
                                  shuffle=False)

        train_dataset = get_representations(bert_model, train_data, device)
        val_dataset = get_representations(bert_model, val_data, device)

        data_train, y_train, g_train = get_data(train_dataset)
        data_val, y_val, g_val = get_data(val_dataset)

        # print(nearsub(args, data_train, y_train, data_val, y_val))

        F1, y_pred_task = evaluate_mlp(data_train, y_train, data_val, y_val)
        print(f"Stage {stage} Task: F1 - {F1}")
        
        bias_F1, y_pred_bias = evaluate_mlp(data_train, g_train, data_val, g_val)
        print(f"Bias: F1 - {bias_F1}")
        
        maj_baseline = len(g_val[g_val==0]) / len(g_val)
        leakage = abs(max(100-bias_F1, bias_F1) - 100 * max(maj_baseline, 1- maj_baseline))
        print(f"Majority Baseline: {maj_baseline} Leakage: {leakage}")

        dp = get_demographic_parity(y_pred_task, g_val)
        tpr = TPR_rms(y_val, y_pred_task, g_val)
        print(f"DP: {dp} TPR: {tpr}")

    elif stage > 0:
        old_train = dataset.get_cumulative_group(stage)
        old_train_data = DataLoader(BatchData(old_train[0], old_train[1], old_train[2]),
                                    batch_size=batch_size,
                                    shuffle=True)
        
        old_train_dataset = get_representations(bert_model, old_train_data, device)
        old_data_train, old_y_train, old_g_train = get_data(old_train_dataset)
        
        _, _, old_test = dataset.get_next_group(stage)
        old_val_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
                                  batch_size=batch_size,
                                  shuffle=False)
        
        old_val_dataset = get_representations(bert_model, old_val_data, device)
        data_old_val, y_old_val, g_old_val = get_data(old_val_dataset)
        
        
        F1, y_pred_task = evaluate_mlp(old_data_train, old_y_train, data_old_val, y_old_val)
        print(f"Task: F1 - {F1}")
        
        bias_F1, y_pred_bias = evaluate_mlp(old_data_train, old_g_train, data_old_val, g_old_val)
        print(f"Bias: F1 - {bias_F1}")
        
        maj_baseline = len(g_old_val[g_old_val==0]) / len(g_old_val)
        leakage = abs(max(100-bias_F1, bias_F1) - 100 * max(maj_baseline, 1- maj_baseline))
        print(f"Majority Baseline: {maj_baseline} Leakage: {leakage}")

        dp = get_demographic_parity(y_pred_task, g_old_val)
        tpr = TPR_rms(y_old_val, y_pred_task, g_old_val)
        print(f"DP: {dp} TPR: {tpr}")
    print("Stage completed!")
    return F1, bias_F1, leakage, dp, tpr


def probe_past(model_dir, dataset, stage, device, batch_size=32):
    if stage == 0:
        return 
    
    print(f"Starting stage {stage} ...")
    print("Loading model ...", end="\r")
    bert_model = torch.load(os.path.join(model_dir, f"bert-{stage}.pb"))
    bert_model.to(device)
    
    
    netG = torch.load(os.path.join(model_dir, f"netG-{stage}.pb"))
    netG.to(device)
    print("Model loaded!")
    
    train, test, old_test = dataset.get_next_group(stage)
    old_train = dataset.get_cumulative_group(stage)
    
    
    old_train_data = DataLoader(BatchData(old_train[0], old_train[1], old_train[2]),
                                batch_size=batch_size,
                                shuffle=True)
    old_test_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
                              batch_size=batch_size,
                              shuffle=False)
    
    
    train_dataset = get_representations(netG, bert_model, old_train_data, device)
    test_dataset = get_representations(netG, bert_model, old_test_data, device)
    
    
    data_train, g_train = get_data(train_dataset)
    data_val, g_val = get_data(test_dataset)
    
    
    clf_mlp = train_mlp(data_train, g_train)
    F1, P, R = evaluate_mlp(clf_mlp, data_val, g_val)
    print(f"Stage {stage} Old Validation leakage: F1 - {F1} P - {P} R - {R}")
    
    print("Stage completed!")

# +
# if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--batch_size",
                    default=128,
                    type=int,
                    help="Batch Size.")
parser.add_argument("--total_stages",
                    default=6,
                    type=int,
                    help="Total number of stages of training.")
parser.add_argument("--dataset_cache_path",
                    default="../../data/bios.pkl",
                    type=str,
                    help="Bios dataset cache path.")
parser.add_argument("--model_dir",
                    default='../../model/adversarial/',
                    type=str,
                    help="Saved model path.")
parser.add_argument("--device",
                    default="cuda:3",
                    type=str,
                    help="GPU device ID.")
parser.add_argument('--n_comp', 
                    type=int, 
                    default=30, 
                    help='number of components for PCA (default: 30)')
args = parser.parse_args("")

torch.manual_seed(args.seed)
print('Loading data...', end="\r")
bios = load(args.dataset_cache_path)
print('Data Loaded!')
# -

print(args.model_dir)

F1s, bias_F1s, leakages, dps, tprs = [], [], [], [], []
for stage in range(args.total_stages):
    F1, bias_F1, leakage, dp, tpr = probe(args, bios, stage, args.device, args.batch_size)
    F1s.append(F1)
    bias_F1s.append(bias_F1)
    leakages.append(leakage)
    dps.append(dp)
    tprs.append(tpr)

F1s = np.array(F1s)
bias_F1s = np.array(bias_F1s)
leakages = np.array(leakages)
dps = np.array(dps)
tprs = np.array(tprs)

print(f"Avg. old task accuracy: {np.array(F1s).mean()}")
print(f"Avg. old task leakage: {np.array(leakages).mean()}")
print(f"Avg. old task DP: {np.array(dps).mean()}")
print(f"Avg. old task TPR: {np.array(tprs).mean()}")

# +
# dataset = bios
# stage = 5

# bert_model = torch.load(os.path.join(args.model_dir, f"bert-eigen-{stage}.pb"))
# bert_model.to(args.device)


# netG = torch.load(os.path.join(args.model_dir, f"netG-eigen-{stage}.pb"))
# netG.to(args.device)
# print("Model loaded!")

# train, test, old_test = dataset.get_next_group(stage)

# train_data = DataLoader(BatchData(train[0], train[1], train[2]),
#                         batch_size=args.batch_size,
#                         shuffle=True)
# val_data = DataLoader(BatchData(test[0], test[1], test[2]),
#                       batch_size=args.batch_size,
#                       shuffle=False)
# old_val_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
#                           batch_size=args.batch_size,
#                           shuffle=False)

# train_dataset = get_representations(netG, bert_model, train_data, args.device)
# val_dataset = get_representations(netG, bert_model, val_data, args.device)
# old_val_dataset = get_representations(netG, bert_model, old_val_data, args.device)

# data_train, y_train, g_train = get_data(train_dataset)
# data_val, y_val, g_val = get_data(val_dataset)
# data_old_val, y_old_val, g_old_val = get_data(old_val_dataset)

# +
# old_train = dataset.get_cumulative_group(5)
# train, test, old_test = dataset.get_next_group(5)

# old_train_data = DataLoader(BatchData(old_train[0], old_train[1], old_train[2]),
#                             batch_size=args.batch_size,
#                             shuffle=True)
# old_test_data = DataLoader(BatchData(old_test[0], old_test[1], old_test[2]),
#                           batch_size=args.batch_size,
#                           shuffle=False)

# +
# old_train_dataset = get_representations(netG, bert_model, old_train_data, args.device)
# old_test_dataset = get_representations(netG, bert_model, old_test_data, args.device)
        
# old_data_train, old_y_train, old_g_train = get_data(old_train_dataset)
# old_data_val, old_y_val, old_g_val = get_data(old_test_dataset)

# +
# nearsub(args, old_data_train, old_y_train, old_data_val, old_y_val)

# +
# nearsub(args, old_data_train, old_g_train, old_data_val, old_g_val)
