import numpy as np

from collections import Counter, defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_mlp(x_train, y_train, x_test, y_test):
    """
    Evaluates MLPScore for prediction on a task
    """

    clf = MLPClassifier(max_iter=5, verbose=False)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    F1 = f1_score(y_pred, y_test, average="micro") * 100
    P = precision_score(y_pred, y_test, average="micro") * 100
    R = recall_score(y_pred, y_test, average="micro") * 100
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
    y_main, y_hat_main, y_protected = np.array(y_main), np.array(
        y_hat_main), np.array(y_protected)

    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals["y:{}".format(label)]["p:{}".format(i)] = (
                y_label == y_hat_label).mean()

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
