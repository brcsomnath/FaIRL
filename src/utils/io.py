import torch
import pickle
import numpy as np

from tqdm import tqdm


def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump(content, filename):
    with open(filename, "wb") as f:
        pickle.dump(content, f)


def load_content(content, DATA_IDX=1, Y_IDX=2, Z_IDX=3):
    """
    Forms the data in the format (sentence, y-label, z-label)
    Returns an array of instances in the above format
    """
    dataset = []
    for c in content:
        dataset.append((c[DATA_IDX], c[Y_IDX], c[Z_IDX]))
    return dataset


def set_seed(args):
    """
    Set the random seed for reproducibility
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def encode(X, Y, Z):
    """Converts data into token representations in the form of tensors."""
    encoded_dataset = []
    for x, y, z in zip(X, Y, Z):
        emb = torch.tensor(x)
        encoded_dataset.append((emb, y, z))
    return encoded_dataset

def get_data(dataset):
    data = np.array([d[0] for d in dataset])
    task_label = np.array([d[1] for d in dataset])
    bias_label = np.array([d[2] for d in dataset])
    return data, task_label, bias_label
