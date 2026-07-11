import numpy as np
import numba
from numba import prange
from mpi4py import MPI


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
            np.int32
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

    def halo_exchange(
        self,
        state,
        bool_buffers=None,
        int_buffers=None,
        float_buffers=None
    ):
        """
        Exchanges fields between mpi processors
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
                send_copy_y = send_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_y = send_copy_y_vector
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
                recv_copy_y = recv_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_y = recv_copy_y_vector
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
                send_copy_y = send_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_y = send_copy_y_vector
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
                recv_copy_y = recv_copy_y_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_y = recv_copy_y_vector
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
                send_copy_x = send_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_x = send_copy_x_vector
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
                recv_copy_x = recv_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_x = recv_copy_x_vector
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
                send_copy_x = send_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    send_copy_x = send_copy_x_vector
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
                recv_copy_x = recv_copy_x_scalar
                if (layout_end - layout_start) > 1:
                    recv_copy_x = recv_copy_x_vector
                recv_copy_x(
                    buffer_object.recv_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=0
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


@numba.njit(parallel=True, nogil=True)
def send_copy_y_vector(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            send_buff[y, itr] = field[ind, itr - layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_y_scalar(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        send_buff[y, layout_start] = field[ind]


@numba.njit(parallel=True, nogil=True)
def recv_copy_y_vector(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            field[ind, itr - layout_start] = recv_buff[y, itr]


@numba.njit(parallel=True, nogil=True)
def recv_copy_y_scalar(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        field[ind] = recv_buff[y, layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_x_vector(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            send_buff[x, itr] = field[ind, itr - layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_x_scalar(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        send_buff[x, layout_start] = field[ind]


@numba.njit(parallel=True, nogil=True)
def recv_copy_x_vector(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            field[ind, itr - layout_start] = recv_buff[x, itr]


@numba.njit(parallel=True, nogil=True)
def recv_copy_x_scalar(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        field[ind] = recv_buff[x, layout_start]
