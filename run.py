from LBpy.LBpy import main
import mpi4py


if __name__ == '__main__':
    mpi4py.rc(initialize=False, finalize=False)
    main()
