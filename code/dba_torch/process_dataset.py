import os
import sys
from datetime import date
import shutil
from functools import partial

import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader

from models_gat_sagp import DBA, Encoder, StructureEncoder, Decoder
from graphdata import PairData, PairDataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", default=0, type=int, help="Architecture Name")

args = parser.parse_args()
# device = "cuda:{:d}".format(args.dataset)

# loop through folders and load data
ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 4e6, 6e6, 8e6, 9e6, 1e7]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]
n_slices = 5
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

if args.dataset == 0:
  process_dataset = PairDataset(data_path, ma_list, re_list, aoa_list, "train",
                              n_slices)
elif args.dataset == 1:
  process_dataset = PairDataset(data_path, ma_list, [3e6, 5e6, 7e6], aoa_list, "test",
                              n_slices)
elif args.dataset == 2:
  process_dataset = PairDataset(data_path, [0.3], [3e6], [3],
                              "idev-train", n_slices)
elif args.dataset == 3:
  process_dataset = PairDataset(data_path, [0.6], [6e6], [6], "idev-test",
                              n_slices)