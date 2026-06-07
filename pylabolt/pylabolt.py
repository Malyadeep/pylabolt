from argparse import ArgumentParser
import os
import mpi4py


def main():
    mpi4py.rc(initialize=False, finalize=False)
    parser = ArgumentParser(description="A Lattice Boltzmann Python solver")
    parser.add_argument("-s", "--solver", choices=["fluidLB", "phaseFieldLB",
                        "cgLB"], type=str, help="choice of solver to run")
    parser.add_argument("-b", "--backend", choices=["cpu", "gpu"],
                        default="cpu", type=str, help="choice of backend")
    parser.add_argument("-c", "--cuda", action="store_true", default=False,
                        help="set to run simulation in parallel using CUDA")
    parser.add_argument("-nt", "--n_threads", type=int, default=1,
                        help="Number of threads for OpenMP/CUDA")
    parser.add_argument("--reconstruct", choices=["last", "all", "time", None],
                        default=None, help="Domain reconstruction"
                        )
    parser.add_argument("-t", "--time", type=int, default=0,
                        help="Specify time which is to be reconstructed")
    parser.add_argument("--toVTK", choices=["last", "all", "time", None],
                        default=None, help="Convert output data to VTK format")
    args = parser.parse_args()

    if args.solver == "fluidLB":
        from pylabolt.solvers import fluidLB
        fluidLB.main(args.backend, args.n_threads)
    if args.solver == "phaseFieldLB":
        from pylabolt.solvers import phaseFieldLB
        phaseFieldLB.main(args.backend, args.n_threads)
    # elif args.solver == "cgLB":
    #     from pylabolt.solvers.cgLB import cgLB
    #     cgLB.main(parallelization, n_threads=args.n_threads)

    # if args.reconstruct is not None:
    #     from pylabolt.parallel.MPI_reconstruct import reconstruct
    #     reconstruct(args.reconstruct, args.time)

    if args.toVTK is not None:
        from pylabolt.utils.vtkconvert import toVTK
        toVTK(args.toVTK, args.time)
