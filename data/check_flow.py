import os
import glob

ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]

print("\nMISSING FLOW: \n")
for ma in ma_list:
  for re in re_list:
    for aoa in aoa_list:
      if len(glob.glob(os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data","ma_{:g}/re_{:g}/a_{:d}/flow.vtu".format(ma,re,aoa)))) == 0:
        print("ma_{:g} --- re_{:g} --- a_{:d}".format(ma,re,aoa))

print("------------- \n\nMISSING SLICE: \n")
for ma in ma_list:
  for re in re_list:
    for aoa in aoa_list:
      if len(glob.glob(os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data","ma_{:g}/re_{:g}/a_{:d}/slice*.vtk".format(ma,re,aoa)))) == 0:
        print("ma_{:g} --- re_{:g} --- a_{:d}".format(ma,re,aoa))