import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", default="main_sr.py", type=str, help="file name")
parser.add_argument("--nodes", "-N", default=0, type=int, help="nodes")
parser.add_argument("--tasks", "-n", default=4, type=int, help="sims")
args = parser.parse_args()

file = args.file
number_of_nodes = args.nodes
tasks_per_node = args.tasks

channels = [128]
layers = [1]
omegas = [11]
latent_sizes = [1, 4, 8, 16]
learning_rate = [1e-4, 1e-5]

# mach = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1]
# mach = [0.8395]
# reynolds = [1e5, 1e6, 1e7, 1e8]
# reynolds = [11.72e6]
# alpha = [0, 2, 4, 6, 8, 10, 12]
# alpha = [3.06]

if number_of_nodes == 0:
    total_sims = float(len(channels) * len(learning_rate))
    number_of_nodes = int(-(total_sims // -tasks_per_node))

LIST = []
for c in channels:
    for l in layers:
        for o in omegas:
            for s in latent_sizes:
                for lr in learning_rate:
                    line = (
                        file
                        + " --channels {:g} --layers {:g} --omega {:g} --latent-sz {:g} --learning-rate {:g} --wandb 1".format(
                            c, l, o, s, lr
                        )
                    )
                    LIST.append(line)

LIST2 = []
for node in range(number_of_nodes):
    for sim in range(tasks_per_node):
        if node * tasks_per_node + sim < len(LIST):
            line = (
                "cd /scratch1/07169/wgl/ORNL/dimension-bridging; apptainer exec --nv db.sif /app/jax-env/bin/python code/dbsr/"
                + LIST[node * tasks_per_node + sim]
                + " --gpu-id {:d}".format(sim)
            )
        if node * tasks_per_node + sim >= len(LIST):
            line = " "

        LIST2.append(line)

for node in range(number_of_nodes):
    names = LIST2[node * tasks_per_node : (node + 1) * tasks_per_node]

    with open(r"argument_files" + str(node + 1), "w") as fp:
        for item in names:
            fp.write("%s\n" % item)
    fp.close()
