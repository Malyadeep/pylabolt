from argparse import ArgumentParser
import os
import mpi4py


def main():
    mpi4py.rc(initialize=False, finalize=False)
    parser = ArgumentParser(description='A Lattice Boltzmann Python solver')
    parser.add_argument('-s', '--solver', choices=['fluidLB'], type=str,
                        help='choice of solver to run')
    parser.add_argument('-p', '--parallel', action='store_true', default=False,
                        help='set to run simulation in parallel using OpenMP')
    parser.add_argument('-c', '--cuda', action='store_true', default=False,
                        help='set to run simulation in parallel using CUDA')
    parser.add_argument('-nt', '--n_threads', type=int, default=None,
                        help='Number of threads for OpenMP/CUDA')
    args = parser.parse_args()

    if args.parallel is True:
        parallelization = 'openmp'
        if args.n_threads is None:
            raise RuntimeError("ERROR! Require number of threads!")
            os._exit(1)
    if args.cuda is True:
        parallelization = 'cuda'
        if args.n_threads is None:
            raise RuntimeError("ERROR! Require number of threads!")
            os._exit(1)
    if args.cuda is True and args.parallel is True:
        raise RuntimeError("ERROR! set a single backend for parallelization!")
        os._exit(1)
    elif args.parallel is False and args.cuda is False:
        parallelization = None

    if args.solver == 'fluidLB':
        from LBpy.solvers import fluidLB
        fluidLB.main(parallelization, n_threads=args.n_threads)
