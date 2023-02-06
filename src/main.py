import torch
import argparse
import numpy as np

from utils.io import *
from LwF.trainer import LWFTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch Size.")
    parser.add_argument("--epochs",
                        default=1,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--alpha", default=1, type=float, help="Mixture ratio")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--device",
                        default="cuda:3",
                        type=str,
                        help="GPU device ID.")
    parser.add_argument("--dataset_cache_path",
                        default="../data/bios.pkl",
                        type=str,
                        help="Bios dataset cache path.")
    parser.add_argument("--save_dir",
                        default='../model/finetune/',
                        type=str,
                        help="Tokenized dataset cache path.")
    parser.add_argument("--old_loss",
                        default=False,
                        type=bool,
                        help="Include old loss term during training.")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bios = load(args.dataset_cache_path)

    trainer = LWFTrainer(bios,
                         device=args.device,
                         save_dir=args.save_dir,
                         old_loss=args.old_loss)
    trainer.trainer(batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    alpha=args.alpha)
