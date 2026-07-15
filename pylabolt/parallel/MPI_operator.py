import numpy as np
from mpi4py import MPI

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.MPI_kernels as MPI_kernels_cpu
# import pylabolt.parallel.gpu.MPI_kernels as MPI_kernels_gpu


class HaloBuffer:
    def __init__(
        self,
        state,
        field_dict,
        dtype
    ):
        """
        Container for MPI communication buffers
        Attributes:

        """
        self.field_dict = field_dict
        self.layout = {}
        self.components = 0
        self.offset = 0
        for key in self.field_dict:
            self.components += self.field_dict[key]
            self.register_field(key)
            self.offset += self.field_dict[key]
        self.send_buff_left_right = np.zeros(
            (state.domain.shape[1], self.components),
            dtype=dtype
        )
        self.recv_buff_left_right = np.zeros(
            (state.domain.shape[1], self.components),
            dtype=dtype
        )
        self.send_buff_top_bottom = np.zeros(
            (state.domain.shape[0], self.components),
            dtype=dtype
        )
        self.recv_buff_top_bottom = np.zeros(
            (state.domain.shape[0], self.components),
            dtype=dtype
        )

    def register_field(
        self,
        key
    ):
        """
        Registers field layout in the buffer
        Args:

        Returns:

        """
        self.layout.update({
            key: (self.offset, self.offset + self.field_dict[key])
        })


class MPIOperator:
    def __init__(
        self,
        comm,
        state
    ):
        """
        MPI operator - operations for MPI communication
        Attributes:

        """
        self.comm = comm
        bool_field_dict = {
            "solid": 1,
            "solid_boundary": 1,
            "fluid_boundary": 1
        }
        self.bool_buffer = HaloBuffer(
            state,
            bool_field_dict,
            np.bool_
        )

        int_field_dict = {
            "solid_id": 1
        }
        self.int_buffer = HaloBuffer(
            state,
            int_field_dict,
            int
        )

        float_field_dict = {}
        if state.fluid is True:
            float_field_dict.update({
                "pop_fluid": state.lattice.no_of_directions,
                "velocity": 2,
                "pressure": 1,
                "density": 1
            })
        if state.phase is True:
            float_field_dict.update({
                "pop_phase": state.lattice.no_of_directions,
                "phase_field": 1,
                "grad_phase_field": 2
            })
        self.float_buffer = HaloBuffer(
            state,
            float_field_dict,
            state.control.precision
        )

        self.find_neighbor_ranks(state)

    def find_neighbor_ranks(
        self,
        state
    ):
        """
        Finds and sets neighboring ranks
        Args:

        Returns:

        """
        i_proc = state.domain.i_proc
        j_proc = state.domain.j_proc
        no_of_procs_x = state.domain.no_of_procs_x
        no_of_procs_y = state.domain.no_of_procs_y
        if i_proc == 0 and not state.boundary.x_periodic:
            self.left_rank = None
        else:
            i_proc_nb = (i_proc - 1 + no_of_procs_x) % no_of_procs_x
            self.left_rank = i_proc_nb * no_of_procs_y + j_proc

        if i_proc == no_of_procs_x - 1 and not state.boundary.x_periodic:
            self.right_rank = None
        else:
            i_proc_nb = (i_proc + 1 + no_of_procs_x) % no_of_procs_x
            self.right_rank = i_proc_nb * no_of_procs_y + j_proc

        if j_proc == 0 and not state.boundary.y_periodic:
            self.bottom_rank = None
        else:
            j_proc_nb = (j_proc - 1 + no_of_procs_y) % no_of_procs_y
            self.bottom_rank = i_proc * no_of_procs_y + j_proc_nb

        if j_proc == no_of_procs_y - 1 and not state.boundary.y_periodic:
            self.top_rank = None
        else:
            j_proc_nb = (j_proc + 1 + no_of_procs_y) % no_of_procs_y
            self.top_rank = i_proc * no_of_procs_y + j_proc_nb

    def halo_exchange_cpu(
        self,
        state,
        bool_buffers=None,
        int_buffers=None,
        float_buffers=None
    ):
        """
        Exchanges fields between mpi processors for CPU backend
        Args:

        Returns:

        """
        self.comm.Barrier()
        if bool_buffers is not None:
            buffer_object = self.bool_buffer
            args = (bool_buffers, buffer_object, state)
            if state.domain.i_proc % 2 == 0:
                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)
            else:
                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)

                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

            self.comm.Barrier()
            if state.domain.j_proc % 2 == 0:
                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)
            else:
                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)

                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

        if int_buffers is not None:
            buffer_object = self.int_buffer
            args = (int_buffers, buffer_object, state)
            if state.domain.i_proc % 2 == 0:
                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)
            else:
                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)

                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

            self.comm.Barrier()
            if state.domain.j_proc % 2 == 0:
                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)
            else:
                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)

                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

        if float_buffers is not None:
            buffer_object = self.float_buffer
            args = (float_buffers, buffer_object, state)
            if state.domain.i_proc % 2 == 0:
                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)
            else:
                # ------- Send to right, receive from right ------- #
                self._exchange_right(*args)

                # ------- Send to left, receive from left ------- #
                self._exchange_left(*args)

            self.comm.Barrier()
            if state.domain.j_proc % 2 == 0:
                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)
            else:
                # ------- Send to top, receive from top ------- #
                self._exchange_top(*args)

                # ------- Send to bottom, receive from bottom ------- #
                self._exchange_bottom(*args)

    def _exchange_left(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Exchange left mpi processor boundary
        Args:

        Returns:

        """
        if self.left_rank is not None:
            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                send_copy_y = MPI_kernels_cpu.send_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_y = MPI_kernels_cpu.send_copy_y_vector
                send_copy_y(
                    buffer_object.send_buff_left_right,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=1
                )

            self.comm.Sendrecv(
                sendbuf=buffer_object.send_buff_left_right,
                dest=self.left_rank,
                sendtag=0,
                recvbuf=buffer_object.recv_buff_left_right,
                source=self.left_rank,
                recvtag=0
            )

            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                recv_copy_y = MPI_kernels_cpu.recv_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_y = MPI_kernels_cpu.recv_copy_y_vector
                recv_copy_y(
                    buffer_object.recv_buff_left_right,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=0
                )

    def _exchange_right(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Exchange right mpi processor boundary
        Args:

        Returns:

        """
        if self.right_rank is not None:
            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                send_copy_y = MPI_kernels_cpu.send_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_y = MPI_kernels_cpu.send_copy_y_vector
                send_copy_y(
                    buffer_object.send_buff_left_right,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=(state.domain.shape[0] - 2)
                )

            self.comm.Sendrecv(
                sendbuf=buffer_object.send_buff_left_right,
                dest=self.right_rank,
                sendtag=0,
                recvbuf=buffer_object.recv_buff_left_right,
                source=self.right_rank,
                recvtag=0
            )

            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                recv_copy_y = MPI_kernels_cpu.recv_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_y = MPI_kernels_cpu.recv_copy_y_vector
                recv_copy_y(
                    buffer_object.recv_buff_left_right,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=(state.domain.shape[0] - 1)
                )

    def _exchange_top(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Exchange top mpi processor boundary
        Args:

        Returns:

        """
        if self.top_rank is not None:
            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                send_copy_x = MPI_kernels_cpu.send_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_x = MPI_kernels_cpu.send_copy_x_vector
                send_copy_x(
                    buffer_object.send_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=(state.domain.shape[1] - 2)
                )

            self.comm.Sendrecv(
                sendbuf=buffer_object.send_buff_top_bottom,
                dest=self.top_rank,
                sendtag=0,
                recvbuf=buffer_object.recv_buff_top_bottom,
                source=self.top_rank,
                recvtag=0
            )

            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                recv_copy_x = MPI_kernels_cpu.recv_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_x = MPI_kernels_cpu.recv_copy_x_vector
                recv_copy_x(
                    buffer_object.recv_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=(state.domain.shape[1] - 1)
                )

    def _exchange_bottom(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Exchange top mpi processor boundary
        Args:

        Returns:

        """
        if self.bottom_rank is not None:
            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                send_copy_x = MPI_kernels_cpu.send_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_x = MPI_kernels_cpu.send_copy_x_vector
                send_copy_x(
                    buffer_object.send_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=1
                )

            self.comm.Sendrecv(
                sendbuf=buffer_object.send_buff_top_bottom,
                dest=self.bottom_rank,
                sendtag=0,
                recvbuf=buffer_object.recv_buff_top_bottom,
                source=self.bottom_rank,
                recvtag=0
            )

            for field_name in buffer_names:
                layout_start, layout_end = \
                    buffer_object.layout[field_name]
                field_to_copy = getattr(state.fields, field_name)
                recv_copy_x = MPI_kernels_cpu.recv_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_x = MPI_kernels_cpu.recv_copy_x_vector
                recv_copy_x(
                    buffer_object.recv_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=0
                )

    def halo_exchange_gpu(
        self,
        state,
        bool_buffers=None,
        int_buffers=None,
        float_buffers=None
    ):
        """
        Supports only single GPU execution
        Thus, only performs periodic wrapping of ghost node data
        No MPI communitation involved, only copying kernels
        Args:

        Returns:

        """
        if state.boundary.x_periodic:
            if bool_buffers is not None:
                buffer_object = self.bool_buffer
                args = (bool_buffers, buffer_object, state)
                self._exchange_horizontal_gpu(*args)
            if int_buffers is not None:
                buffer_object = self.int_buffer
                args = (int_buffers, buffer_object, state)
                self._exchange_horizontal_gpu(*args)
            if float_buffers is not None:
                buffer_object = self.float_buffer
                args = (float_buffers, buffer_object, state)
                self._exchange_horizontal_gpu(*args)

        if state.boundary.y_periodic:
            if bool_buffers is not None:
                buffer_object = self.bool_buffer
                args = (bool_buffers, buffer_object, state)
                self._exchange_vertical_gpu(*args)
            if int_buffers is not None:
                buffer_object = self.int_buffer
                args = (int_buffers, buffer_object, state)
                self._exchange_vertical_gpu(*args)
            if float_buffers is not None:
                buffer_object = self.float_buffer
                args = (float_buffers, buffer_object, state)
                self._exchange_vertical_gpu(*args)

    def _exchange_horizontal_gpu(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Periodic boundary copying in horizontal direction on GPU
        Args:

        Returns:

        """
        for field_name in buffer_names:
            layout_start, layout_end = \
                buffer_object.layout[field_name]
            field_to_copy = getattr(state.fields, field_name)
            copy_y = MPI_kernels_gpu.copy_y_scalar
            if (layout_end - layout_start) > 1:
                copy_y = MPI_kernels_gpu.copy_y_vector
            copy_y(
                buffer_object.send_buff_left_right,
                field_to_copy,
                layout_start,
                layout_end,
                state.domain.shape,
                x=1
            )

    def _exchange_vertical_gpu(
        self,
        buffer_names,
        buffer_object,
        state
    ):
        """
        Periodic boundary copying in vertical direction on GPU
        Args:

        Returns:

        """
        for field_name in buffer_names:
            layout_start, layout_end = \
                buffer_object.layout[field_name]
            field_to_copy = getattr(state.fields, field_name)
            copy_y = MPI_kernels_gpu.copy_x_scalar
            if (layout_end - layout_start) > 1:
                copy_y = MPI_kernels_gpu.copy_x_vector
            copy_y(
                buffer_object.send_buff_left_right,
                field_to_copy,
                layout_start,
                layout_end,
                state.domain.shape,
                x=1
            )

    def reduce(
        self,
        local_array,
        operation="sum"
    ):
        """
        Performs specified reduction operation across all
        MPI ranks
        Args:

        Returns:

        """
        global_array = np.zeros_like(local_array)
        if operation == "sum":
            self.comm.Allreduce(local_array, global_array, op=MPI.SUM)
        return global_array

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile MPI operator kernels
        Args:

        Returns:

        """
        for buffer, field_name, compilation_type in [
            (self.bool_buffer, "solid", "bool_scalar"),
            (self.int_buffer, "solid_id", "int_scalar"),
            (self.float_buffer, "density", "float_scalar"),
            (self.float_buffer, "pop_fluid", "float_vector")
        ]:
            layout_start, layout_end = \
                buffer.layout[field_name]
            field_to_copy = getattr(state.fields, field_name)
            args = (
                buffer.send_buff_top_bottom,
                field_to_copy,
                layout_start,
                layout_end,
                state.domain.shape
            )
            compile_args = backend.make_compile_args(args)
            send_copy_x = MPI_kernels_cpu.send_copy_x_scalar
            recv_copy_x = MPI_kernels_cpu.recv_copy_x_scalar
            if (layout_end - layout_start) > 1:
                send_copy_x = MPI_kernels_cpu.send_copy_x_vector
                recv_copy_x = MPI_kernels_cpu.recv_copy_x_vector
            send_copy_x(*compile_args, y=0)
            recv_copy_x(*compile_args, y=0)

            layout_start, layout_end = \
                buffer.layout[field_name]
            field_to_copy = getattr(state.fields, field_name)
            args = (
                buffer.send_buff_left_right,
                field_to_copy,
                layout_start,
                layout_end,
                state.domain.shape
            )
            compile_args = backend.make_compile_args(args)
            send_copy_y = MPI_kernels_cpu.send_copy_y_scalar
            recv_copy_y = MPI_kernels_cpu.recv_copy_y_scalar
            if (layout_end - layout_start) > 1:
                send_copy_y = MPI_kernels_cpu.send_copy_y_vector
                recv_copy_y = MPI_kernels_cpu.recv_copy_y_vector
            send_copy_y(*compile_args, x=0)
            recv_copy_y(*compile_args, x=0)

        self.kernel_signatures = {
            MPI_kernels_cpu.send_copy_x_scalar.__name__:
                set(MPI_kernels_cpu.send_copy_x_scalar.signatures),
            MPI_kernels_cpu.send_copy_x_vector.__name__:
                set(MPI_kernels_cpu.send_copy_x_vector.signatures),
            MPI_kernels_cpu.send_copy_y_scalar.__name__:
                set(MPI_kernels_cpu.send_copy_y_scalar.signatures),
            MPI_kernels_cpu.send_copy_y_vector.__name__:
                set(MPI_kernels_cpu.send_copy_y_vector.signatures),
            MPI_kernels_cpu.recv_copy_x_scalar.__name__:
                set(MPI_kernels_cpu.recv_copy_x_scalar.signatures),
            MPI_kernels_cpu.recv_copy_x_vector.__name__:
                set(MPI_kernels_cpu.recv_copy_x_vector.signatures),
            MPI_kernels_cpu.recv_copy_y_scalar.__name__:
                set(MPI_kernels_cpu.recv_copy_y_scalar.signatures),
            MPI_kernels_cpu.recv_copy_y_vector.__name__:
                set(MPI_kernels_cpu.recv_copy_y_vector.signatures),
        }

        # for item in self.kernel_signatures:
        #     print(item, self.kernel_signatures[item])

        print_log("Compiled MPI operator",
                  state.domain.mpi_rank, verbose)

    def set_backend(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Sets backend for MPI operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.halo_exchange = self.halo_exchange_cpu
        elif backend.backend_type == "gpu":
            self.halo_exchange = self.halo_exchange_gpu

        print_log("Backend set for MPI operator",
                  state.domain.mpi_rank, verbose)

    def verify_kernel_signatures(
        self,
        state,
        backend,
        verbose=True
    ):
        for kernel_name in self.kernel_signatures:
            kernel = getattr(MPI_kernels_cpu, kernel_name)
            if set(kernel.signatures) != self.kernel_signatures[kernel_name]:
                raise RuntimeError(
                    f"Developer error! {kernel_name} compiled a new signature!"
                )
        print_log("Kernel signatures verified for MPI operator",
                  state.domain.mpi_rank, verbose)
