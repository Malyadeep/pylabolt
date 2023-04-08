from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description='A Lattice Boltzmann Python solver')
    parser.add_argument('-s', '--solve', choices=['fluidLB'], type=str,
                        help='choice of solver to run')
    args = parser.parse_args()

    if args.solve == 'fluidLB':
        from LBpy.solvers import fluidLB
        fluidLB.main()
