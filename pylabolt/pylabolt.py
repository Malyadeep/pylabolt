from argparse import ArgumentParser
import mpi4py


def main():
    mpi4py.rc(initialize=False, finalize=False)
    parser = ArgumentParser(description="A Lattice Boltzmann Python solver")
    parser.add_argument(
        "-s", "--solver", choices=["fluidLB", "phaseFieldLB", "cgLB"],
        type=str, help="choice of solver to run"
    )
    parser.add_argument(
        "-b", "--backend", choices=["cpu", "gpu"], default="cpu",
        type=str, help="choice of backend: CPU/GPU (CUDA)"
    )
    parser.add_argument(
        "-nt", "--n_threads", type=int, default=1,
        help="Number of threads for OpenMP/CUDA"
    )
    parser.add_argument(
        "--reconstruct", choices=["all", "time", None], default=None,
        help="Domain reconstruction"
    )
    parser.add_argument(
        "-t", "--time", type=int, default=0,
        help="Specify time which is to be reconstructed"
    )
    parser.add_argument(
        "--to_vtk", choices=["all", "time", None], default=None,
        help="Convert output data to VTK format"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Run in debug mode"
    )
    args = parser.parse_args()

    """ Run Solver to generate raw output data """
    if args.solver == "fluidLB":
        from pylabolt.solvers import fluidLB
        fluidLB.main(args.backend, args.n_threads, debug_mode=args.debug)
    if args.solver == "phaseFieldLB":
        from pylabolt.solvers import phaseFieldLB
        phaseFieldLB.main(args.backend, args.n_threads, debug_mode=args.debug)
    # elif args.solver == "cgLB":
    #     from pylabolt.solvers.cgLB import cgLB
    #     cgLB.main(parallelization, n_threads=args.n_threads)

    """ Reconstruct raw data from MPI simulations """
    if args.reconstruct is not None:
        from pylabolt.utils.reconstruct import reconstruct_data
        reconstruct_data(args.reconstruct, time_step=args.time)

    """ Convert raw data to VTK format """
    if args.to_vtk is not None:
        from pylabolt.utils.npz2vtk import convert_to_vtk
        convert_to_vtk(args.to_vtk, args.time)
