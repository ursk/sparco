from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()
root = 0

bcast = MPI.COMM_WORLD.Bcast
scatter = MPI.COMM_WORLD.Scatter
gather = MPI.COMM_WORLD.Gather
