import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", default="process_dataset.py", type=str, help="file name")
parser.add_argument("--slices", "-s", default=0, type=int, help="slices")
parser.add_argument("--nodes", "-N", default=0, type=int, help="nodes")
parser.add_argument("--tasks", "-n", default=4, type=int, help="sims")
args = parser.parse_args()

file = args.file
number_of_nodes = args.nodes
tasks_per_node = args.tasks

datasets = [i for i in range(12)]

LIST=[]
for ds in datasets:
  line = file + ' --dataset {:d} --slices {:d}'.format(
      ds, args.slices)
  LIST.append(line)

LIST2 = []
for node in range(number_of_nodes):
  for sim in range(tasks_per_node):
    if node*tasks_per_node + sim < len(LIST):
      line = 'cd /scratch1/07169/wgl/ORNL/dimension-bridging; apptainer exec --nv db.sif /app/jax-env/bin/python code/dbsr/' + LIST[
          node*tasks_per_node + sim]
    if node*tasks_per_node + sim >= len(LIST):
      line = ' '

    LIST2.append(line)

for node in range(number_of_nodes):
  names = LIST2[node*tasks_per_node:(node+1)*tasks_per_node]

  with open(r'process_argument_files' + str(node + 1), 'w') as fp:
    for item in names:
      fp.write("%s\n" % item)
  fp.close()
