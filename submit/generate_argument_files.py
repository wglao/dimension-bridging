import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", "-f", default="main.py", type=str, help="file name")
parser.add_argument("--nodes", "-N", default=0, type=int, help="nodes")
parser.add_argument("--tasks", "-n", default=3, type=int, help="sims")
args = parser.parse_args()

file = args.file
number_of_nodes = args.nodes
tasks_per_node = args.tasks

# channels = [10,25,50,100,200]
channels = [50]
# channels = [10]
# latent_sz = [10,25,50,100,200]
latent_sz = [50]
# latent_sz = [10]
# pooling = [1,2,3,4,5]
pooling = [1]
# lam_2d = [0.01,0.1,1,10,100]
lam_2d = [0.1,1,10]
# lam_2d = [1]
# lam_dp = [0.01,0.1,1,10,100]
lam_dp = [1]

# mach = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1]
# mach = [0.8395]
# reynolds = [1e5, 1e6, 1e7, 1e8]
# reynolds = [11.72e6]
# alpha = [0, 2, 4, 6, 8, 10, 12]
# alpha = [3.06]

if number_of_nodes == 0:
  total_sims = float(
      len(channels)*len(latent_sz)*len(pooling)*len(lam_2d)*len(lam_dp))
  number_of_nodes = int(-(total_sims // -tasks_per_node))

LIST = []
for c in channels:
  for s in latent_sz:
    for p in pooling:
      for l2d in lam_2d:
        for ldp in lam_dp:
          line = file + ' --channels {:g} --latent-sz {:g} --pooling-layers {:g} --lambda-2d {:g} --lambda-dp {:g} --wandb 1'.format(
              c, s, p, l2d, ldp)
          LIST.append(line)

LIST2 = []
for node in range(number_of_nodes):
  for sim in range(tasks_per_node):
    if node*tasks_per_node + sim < len(LIST):
      line = 'cd /scratch/07169/wgl/ORNL/dimension-bridging; apptainer exec --nv db.sif /app/jax-env/bin/python code/dba/' + LIST[
          node*tasks_per_node + sim] + " --gpu-id {:d}".format(sim)
    if node*tasks_per_node + sim >= len(LIST):
      line = ' '

    LIST2.append(line)

for node in range(number_of_nodes):
  names = LIST2[node*tasks_per_node:(node+1)*tasks_per_node]

  with open(r'argument_files' + str(node + 1), 'w') as fp:
    for item in names:
      fp.write("%s\n" % item)
  fp.close()
