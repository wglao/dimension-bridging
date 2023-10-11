import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", default="postproc.py", type=str, help="file name")
parser.add_argument("--nodes", "-N", default=0, type=int, help="nodes")
parser.add_argument(
    "--tasks", "-n", default=56, type=int, help="sims")
args = parser.parse_args()

file = args.file
number_of_nodes = args.nodes
tasks_per_node = args.tasks

ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]

if number_of_nodes == 0:
  total_sims = float(len(ma_list)*len(re_list)*len(aoa_list))
  number_of_nodes = int(np.ceil(total_sims / tasks_per_node))

pvpy = "/opt/apps/intel19/paraview/5.8.1/bin/pvpython"

LIST = []
for m in ma_list:
  for r in re_list:
    for a in aoa_list:
      line = pvpy + ' ' + file + ' --ma ' + str(
                          m) + ' --re ' + str(r) + ' --a ' + str(
                              a)
      LIST.append(line)

LIST2 = []
for node in range(number_of_nodes):
  for sim in range(tasks_per_node):
    if node*tasks_per_node + sim < len(LIST):
      line = 'cd /scratch1/07169/wgl/ORNL/dimension-bridging/data; ' + LIST[
          node*tasks_per_node + sim]
    if node*tasks_per_node + sim >= len(LIST):
      line = ' '

    LIST2.append(line)

for node in range(number_of_nodes):
  names = LIST2[node*tasks_per_node:(node+1)*tasks_per_node]

  with open(r'argument_files' + str(node + 1), 'w') as fp:
    for item in names:
      fp.write("%s\n" % item)
  fp.close()
