test:
	pytest -m "not mpi"

test-mpi:
	mpirun -np 2 pytest -m mpi --with-mpi
	mpirun -np 4 pytest -m mpi --with-mpi
	mpirun -np 6 pytest -m mpi --with-mpi
	mpirun -np 8 pytest -m mpi --with-mpi
	mpirun -np 9 pytest -m mpi --with-mpi