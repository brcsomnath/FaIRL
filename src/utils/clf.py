from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def train_mlp(x_train, y_train):
    clf = MLPClassifier(max_iter=5)

    clf.fit(x_train, y_train)
    return clf


def evaluate_mlp(clf, x_test, y_test):

    y_pred = clf.predict(x_test)

    F1 = f1_score(y_pred, y_test, average="micro") * 100
    P = precision_score(y_pred, y_test, average="micro") * 100
    R = recall_score(y_pred, y_test, average="micro") * 100
    return F1, P, R
