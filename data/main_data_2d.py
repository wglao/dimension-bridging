import argparse
import os
import sys
import subprocess

import numpy as np
import su2
import scipy as sp

parser = argparse.ArgumentParser()
parser.add_argument("--ma", default=0.8395, type=float)
parser.add_argument("--re", default=11.72e6, type=float)
parser.add_argument("--a", default=3.06, type=float)
# parser.add_argument("-f", "--file", dest="filename",
#                   help="read config from FILE", metavar="FILE")

parser.add_argument("-n", "--partitions", dest="partitions", default=1,
                  help="number of PARTITIONS", metavar="PARTITIONS")     # 1 proc to run script, 7 to run sim
parser.add_argument("-c", "--compute", dest="compute", default="True",
                  help="COMPUTE direct and adjoint problem", metavar="COMPUTE")

parser.add_argument("--nZone", dest="nZone", default=1, help="Define the number of ZONES", metavar="NZONE")
parser.add_argument("--fem", dest="fem", default="False", help="Launch the FEM driver (General driver)", metavar="FEM")
parser.add_argument("--harmonic_balance", dest="harmonic_balance", default="False",
                help="Launch the Harmonic Balance (HB) driver", metavar="HB")
args = parser.parse_args()

# 3D case
cfg_file = "onera_d.cfg"

ma = float(args.ma)
re = float(args.re)
a = float(args.a)

partitions = int(args.partitions)
compute = bool(args.compute)

args.nZone = int( args.nZone )
args.fem = args.fem.upper() == 'TRUE'
args.harmonic_balance = args.harmonic_balance.upper() == 'TRUE'

from mpi4py import MPI   # use mpi4py for parallel run (also valid for serial)
comm = MPI.COMM_WORLD

# check/create directory
# ma_path = "ma_{:g}".format(ma)
# if not os.path.isdir(ma_path):
#   os.mkdir(ma_path)

# re_path = os.path.join(ma_path,"re_{:g}".format(re))
# if not os.path.isdir(re_path):
#   os.mkdir(re_path)

# a_path = os.path.join(re_path,"a_{:g}".format(a))
# if not os.path.isdir(a_path):
#   os.mkdir(a_path)

# Config
config = su2.io.Config(cfg_file)
config.NUMBER_PART = partitions
config.MACH_NUMBER = ma
config.REYNOLDS_NUMBER = re
config.AOA = a

if config.SOLVER == "MULTIPHYSICS":
  print("Parallel computation script not compatible with MULTIPHYSICS solver.")
  exit(1)

cfg_run = "run_" + cfg_file
config.dump(cfg_run)

# # os.system("mpirun -n {:d} SU2_CFD ".format(partitions) + cfg_file)
# # su2.run.run_command("ibrun -n {:d} SU2_CFD ".format(partitions) + cfg_file)
# if (args.nZone == 1):
#   SU2Driver = su2.pysu2.CSinglezoneDriver(cfg_run, args.nZone, comm)
# elif args.harmonic_balance:
#   SU2Driver = su2.pysu2.CHBDriver(cfg_run, args.nZone, comm)
# elif (args.nZone >= 2):
#   SU2Driver = su2.pysu2.CMultizoneDriver(cfg_run, args.nZone, comm)
# else:
#   SU2Driver = su2.pysu2.CSinglezoneDriver(cfg_run, args.nZone, comm)

# import pdb; pdb.set_trace()
# # Launch the solver for the entire computation
# SU2Driver.StartSolver()

# # Postprocess the solver and exit cleanly
# SU2Driver.Postprocessing()

# if SU2Driver != None:
#   del SU2Driver
