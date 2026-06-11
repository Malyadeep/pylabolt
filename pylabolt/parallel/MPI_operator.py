import numpy as np
import numba
from numba import prange


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
        self.dict = field_dict
        self.layout = {}
        self.components = 0
        self.offset = 0
        for key in self.dict:
            self.components += self.dict[key]
            self.register_field(key)
            self.offset += self.dict[key]
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
            key: (self.offset, self.offset + self.dict[key])
        })


class MPIOperator:
    def __init__(
        self,
        state,
        comm
    ):
        """
        MPI operator - operations for MPI communication
        Attributes:

        """
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
        comm,
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
        if bool_buffers is not None:
            # ------- Send to left, receive from left ------- #
            if self.left_rank is not None:
                for field_name in bool_buffers:
                    layout_start, layout_end = \
                        self.bool_buffer.layout[field_name]
                    field_to_copy = getattr(state.fields, field_name)
                    send_copy_y(
                        self.bool_buffer.send_buff_left_right,
                        field_to_copy,
                        layout_start,
                        layout_end,
                        state.domain.shape,
                        x=1
                    )

                comm.Sendrecv(
                    sendbuf=self.bool_buffer.send_buff_left_right,
                    dest=self.left_rank,
                    sendtag=0,
                    recvbuf=self.bool_buffer.recv_buff_left_right,
                    source=self.left_rank,
                    recvtag=0
                )


@numba.njit
def send_copy_y(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=1
):
    if (layout_end - layout_start) > 1:
        for y in prange(0, shape[1]):
            ind = x * shape[1] + y
            for itr in range(layout_start, layout_end):
                send_buff[y, itr] = field[ind, itr - layout_start]
    else:
        for y in prange(0, shape[1]):
            ind = x * shape[1] + y
            send_buff[y, layout_start] = field[ind]


def recv_copy():
    pass
