import os
import sys
from datetime import date
import shutil
from functools import partial

import numpy as np
import torch

from graphdata import PairData, PairDataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=0, type=int, help="Dataset to Process")
parser.add_argument("--slices", default=1, type=int, help="Slices to Process")

args = parser.parse_args()
# device = "cuda:{:d}".format(args.dataset)

# loop through folders and load data
# ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
train_ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45 0.5, 0.55, 0.6]
test_ma_list = [0.65, 0.7, 0.75, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
train_re_list = [2e6, 3e6, 4e6, 5e6, 6e6]
test_re_list = [7e6, 8e6, 9e6]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]
train_aoa_list = [-6, -5, -4, -3, 3, 4, 5, 6]
test_aoa_list = [-9, -8, -7, 7, 8, 9]
n_slices = args.slices
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

if args.dataset == 0:
    process_dataset = PairDataset(
        data_path, train_ma_list, train_re_list, train_aoa_list, "train_coarse_{:d}s".format(n_slices), n_slices
    )
elif args.dataset == 1:
    process_dataset = PairDataset(
        data_path, test_ma_list, test_re_list, test_aoa_list, "test_coarse_{:d}s".format(n_slices), n_slices
    )

if args.dataset == 2:
    process_dataset = PairDataset(
        data_path, train_ma_list, train_re_list, train_aoa_list, "train_fine_{:d}s".format(n_slices), n_slices
    )
elif args.dataset == 3:
    process_dataset = PairDataset(
        data_path, test_ma_list, test_re_list, test_aoa_list, "test_fine_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 4:
    process_dataset = PairDataset(
        data_path, [0.3, 0.4], [3e6, 4e6], [3, 4], "idev-train_coarse_{:d}s".format(n_slices), n_slices
    )
elif args.dataset == 5:
    process_dataset = PairDataset(
        data_path, [0.5, 0.6], [5e6, 6e6], [5, 6], "idev-test_coarse_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 6:
    process_dataset = PairDataset(
        data_path, [0.3, 0.4], [3e6, 4e6], [3, 4], "idev-train_fine_{:d}s".format(n_slices), n_slices
    )
elif args.dataset == 7:
    process_dataset = PairDataset(
        data_path, [0.5, 0.6], [5e6, 6e6], [5, 6], "idev-test_fine_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 8:
    process_dataset = PairDataset(
        data_path, [0.3], [3e6], [3], "recon3_coarse_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 9:
    process_dataset = PairDataset(
        data_path, [0.8], [8e6], [8], "recon8_coarse_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 10:
    process_dataset = PairDataset(
        data_path, [0.3], [3e6], [3], "recon3_fine_{:d}s".format(n_slices), n_slices
    )

elif args.dataset == 11:
    process_dataset = PairDataset(
        data_path, [0.8], [8e6], [8], "recon8_fine_{:d}s".format(n_slices), n_slices
    )