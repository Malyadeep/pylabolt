import pytest
import numpy as np
from mpi4py import MPI


from pylabolt.parallel.MPI_operator import (
    MPIOperator,
    send_copy_x_scalar,
    send_copy_y_scalar,
    send_copy_x_vector,
    send_copy_y_vector,
    recv_copy_x_scalar,
    recv_copy_x_vector,
    recv_copy_y_scalar,
    recv_copy_y_vector
)
from pylabolt.parallel.domain import Domain
from factories import (
    make_mesh,
    make_control,
    make_decompose_dict,
    make_fields,
    make_lattice,
    make_simulation,
    make_boundary
)


comm = MPI.COMM_WORLD


class DummyState:
    def __init__(
        self,
        comm,
        simulation,
        fluid=False,
        phase=False
    ):
        self.mesh = make_mesh((100, 100))
        self.domain = Domain(
            simulation,
            self.mesh,
            comm
        )
        self.boundary = make_boundary()
        self.control = make_control()
        self.lattice = make_lattice(9)
        self.fields = make_fields(self.domain, self.control)
        self.fluid = fluid
        self.phase = phase


class TestMPIOperatorInitialization:
    def test_bool_buffer(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation
        )
        bool_field_dict = {
            "solid": 1,
            "solid_boundary": 1,
            "fluid_boundary": 1
        }
        layout = {
            "solid": (0, 1),
            "solid_boundary": (1, 2),
            "fluid_boundary": (2, 3)
        }
        total_components = layout["fluid_boundary"][1]
        mpi_operator = MPIOperator(comm, state)
        assert mpi_operator.bool_buffer.field_dict == bool_field_dict
        assert mpi_operator.bool_buffer.components == total_components
        assert mpi_operator.bool_buffer.layout == layout
        assert (
            mpi_operator.bool_buffer.send_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.bool_buffer.recv_buff_left_right.shape ==
            (state.domain.shape[1], len(bool_field_dict))
        )
        assert (
            mpi_operator.bool_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )
        assert (
            mpi_operator.bool_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )

    def test_int_buffer(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation
        )
        int_field_dict = {
            "solid_id": 1
        }
        layout = {
            "solid_id": (0, 1)
        }
        total_components = layout["solid_id"][1]
        mpi_operator = MPIOperator(comm, state)
        assert mpi_operator.int_buffer.field_dict == int_field_dict
        assert mpi_operator.int_buffer.components == len(int_field_dict)
        assert mpi_operator.int_buffer.layout == layout
        assert (
            mpi_operator.int_buffer.send_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.int_buffer.recv_buff_left_right.shape ==
            (state.domain.shape[1], len(int_field_dict))
        )
        assert (
            mpi_operator.int_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )
        assert (
            mpi_operator.int_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )

    def test_fluid_float_buffer(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True
        )
        float_field_dict = {
            "pop_fluid": state.lattice.no_of_directions,
            "velocity": 2,
            "pressure": 1,
            "density": 1
        }
        layout = {
            "pop_fluid": (0, state.lattice.no_of_directions),
            "velocity": (state.lattice.no_of_directions,
                         state.lattice.no_of_directions + 2),
            "pressure": (state.lattice.no_of_directions + 2,
                         state.lattice.no_of_directions + 3),
            "density": (state.lattice.no_of_directions + 3,
                        state.lattice.no_of_directions + 4)
        }
        total_components = layout["density"][1]
        mpi_operator = MPIOperator(comm, state)
        assert mpi_operator.float_buffer.field_dict == float_field_dict
        assert mpi_operator.float_buffer.components == total_components
        assert mpi_operator.float_buffer.layout == layout
        assert (
            mpi_operator.float_buffer.send_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.float_buffer.recv_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.float_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )
        assert (
            mpi_operator.float_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )

    def test_phase_float_buffer(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        float_field_dict = {
            "pop_fluid": state.lattice.no_of_directions,
            "velocity": 2,
            "pressure": 1,
            "density": 1,
            "pop_phase": state.lattice.no_of_directions,
            "phase_field": 1,
            "grad_phase_field": 2
        }
        layout = {
            "pop_fluid": (0, state.lattice.no_of_directions),
            "velocity": (state.lattice.no_of_directions,
                         state.lattice.no_of_directions + 2),
            "pressure": (state.lattice.no_of_directions + 2,
                         state.lattice.no_of_directions + 3),
            "density": (state.lattice.no_of_directions + 3,
                        state.lattice.no_of_directions + 4),
            "pop_phase": (state.lattice.no_of_directions + 4,
                          2 * state.lattice.no_of_directions + 4),
            "phase_field": (2 * state.lattice.no_of_directions + 4,
                            2 * state.lattice.no_of_directions + 5),
            "grad_phase_field": (2 * state.lattice.no_of_directions + 5,
                                 2 * state.lattice.no_of_directions + 7)
        }
        total_components = layout["grad_phase_field"][1]
        mpi_operator = MPIOperator(comm, state)
        assert mpi_operator.float_buffer.field_dict == float_field_dict
        assert mpi_operator.float_buffer.components == total_components
        assert mpi_operator.float_buffer.layout == layout
        assert (
            mpi_operator.float_buffer.send_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.float_buffer.recv_buff_left_right.shape ==
            (state.domain.shape[1], total_components)
        )
        assert (
            mpi_operator.float_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )
        assert (
            mpi_operator.float_buffer.send_buff_top_bottom.shape ==
            (state.domain.shape[0], total_components)
        )


class TestCopyFunctions:
    def set_fields_value(self, state):
        state.fields.velocity[:, 0] =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.velocity[:, 1] =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.pressure =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.density[:] =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.phase_field[:] =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.solid[:] =\
            np.bool(np.random.randint(0, 2, state.domain.size))
        state.fields.solid_boundary[:] =\
            np.bool(np.random.randint(0, 2, state.domain.size))
        state.fields.fluid_boundary[:] =\
            np.bool(np.random.randint(0, 2, state.domain.size))
        for k in range(state.lattice.no_of_directions):
            state.fields.pop_fluid[:, k] =\
                np.random.uniform(-1, 1, state.domain.size)
            state.fields.pop_phase[:, k] =\
                np.random.uniform(-1, 1, state.domain.size)
        state.fields.grad_phase_field[:, 0] =\
            np.random.uniform(-1, 1, state.domain.size)
        state.fields.grad_phase_field[:, 1] =\
            np.random.uniform(-1, 1, state.domain.size)

    def compare_buffers(self, buffer_object, state, val=1, direction="x"):
        test_buffer = np.copy(buffer_object.send_buff_left_right)
        for key in list(buffer_object.field_dict.keys()):
            field_to_copy = getattr(state.fields, key)
            field_back_copy = np.copy(field_to_copy)
            no_of_components = buffer_object.field_dict[key]
            layout_start, layout_end = buffer_object.layout[key]
            send_copy_y = send_copy_y_scalar
            recv_copy_y = recv_copy_y_scalar
            send_copy_x = send_copy_x_scalar
            recv_copy_x = recv_copy_x_scalar
            if no_of_components > 1:
                send_copy_y = send_copy_y_vector
                recv_copy_y = recv_copy_y_vector
                send_copy_x = send_copy_x_vector
                recv_copy_x = recv_copy_x_vector
            if direction == "x":
                send_copy_y(
                    buffer_object.send_buff_left_right,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=val
                )
                x = val
                for y in range(0, state.domain.shape[1]):
                    ind = x * state.domain.shape[1] + y
                    if no_of_components == 1:
                        test_buffer[y, layout_start] =\
                            field_to_copy[ind]
                        buffer_object.recv_buff_left_right[y, layout_start] =\
                            field_to_copy[ind]
                    else:
                        for component in range(layout_start, layout_end):
                            test_buffer[y, component] =\
                                field_to_copy[ind, component - layout_start]
                            buffer_object.recv_buff_left_right[y, component] =\
                                field_to_copy[ind, component - layout_start]

                recv_copy_y(
                    buffer_object.recv_buff_left_right,
                    field_back_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    x=val
                )
                assert np.all(field_to_copy == field_back_copy)
            elif direction == "y":
                send_copy_x(
                    buffer_object.send_buff_top_bottom,
                    field_to_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=val
                )
                y = val
                for x in range(0, state.domain.shape[0]):
                    ind = x * state.domain.shape[1] + y
                    if no_of_components == 1:
                        test_buffer[x, layout_start] =\
                            field_to_copy[ind]
                        buffer_object.recv_buff_top_bottom[x, layout_start] =\
                            field_to_copy[ind]
                    else:
                        for component in range(layout_start, layout_end):
                            test_buffer[x, component] =\
                                field_to_copy[ind, component - layout_start]
                            buffer_object.recv_buff_top_bottom[x, component] =\
                                field_to_copy[ind, component - layout_start]

                recv_copy_x(
                    buffer_object.recv_buff_top_bottom,
                    field_back_copy,
                    layout_start,
                    layout_end,
                    state.domain.shape,
                    y=val
                )
                assert np.all(field_to_copy == field_back_copy)
        if direction == "x":
            assert np.all(buffer_object.send_buff_left_right == test_buffer)
        elif direction == "y":
            assert np.all(buffer_object.send_buff_top_bottom == test_buffer)

    def test_send_recv_copy_left(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        self.set_fields_value(state)
        mpi_operator = MPIOperator(comm, state)
        self.compare_buffers(
            mpi_operator.bool_buffer,
            state,
            val=1,
            direction="x"
        )
        self.compare_buffers(
            mpi_operator.int_buffer,
            state,
            val=1,
            direction="x"
        )
        self.compare_buffers(
            mpi_operator.float_buffer,
            state,
            val=1,
            direction="x"
        )

    def test_send_recv_copy_right(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        self.set_fields_value(state)
        mpi_operator = MPIOperator(comm, state)
        self.compare_buffers(
            mpi_operator.bool_buffer,
            state,
            val=(state.domain.shape[0] - 2),
            direction="x"
        )
        self.compare_buffers(
            mpi_operator.int_buffer,
            state,
            val=(state.domain.shape[0] - 2),
            direction="x"
        )
        self.compare_buffers(
            mpi_operator.float_buffer,
            state,
            val=(state.domain.shape[0] - 2),
            direction="x"
        )

    def test_send_recv_copy_bottom(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        self.set_fields_value(state)
        mpi_operator = MPIOperator(comm, state)
        self.compare_buffers(
            mpi_operator.bool_buffer,
            state,
            val=1,
            direction="y"
        )
        self.compare_buffers(
            mpi_operator.int_buffer,
            state,
            val=1,
            direction="y"
        )
        self.compare_buffers(
            mpi_operator.float_buffer,
            state,
            val=1,
            direction="y"
        )

    def test_send_recv_copy_top(self):
        decompose_dict = make_decompose_dict(1, 1)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        self.set_fields_value(state)
        mpi_operator = MPIOperator(comm, state)
        self.compare_buffers(
            mpi_operator.bool_buffer,
            state,
            val=(state.domain.shape[1] - 2),
            direction="y"
        )
        self.compare_buffers(
            mpi_operator.int_buffer,
            state,
            val=(state.domain.shape[1] - 2),
            direction="y"
        )
        self.compare_buffers(
            mpi_operator.float_buffer,
            state,
            val=(state.domain.shape[1] - 2),
            direction="y"
        )


mpi_size = MPI.COMM_WORLD.Get_size()


def valid_decompositions():
    return [
        (nx, mpi_size // nx)
        for nx in (1, mpi_size + 1)
        if mpi_size % nx == 0
    ]


@pytest.fixture(params=valid_decompositions())
def decomposition(request):
    return request.param


@pytest.mark.mpi
@pytest.mark.skipif(
    mpi_size not in [2, 4, 6, 8, 9],
    reason="Requires multiple MPI ranks, supported sizes: 2, 4, 6, 8, 9"
)
class TestHaloExchange:
    def set_fields_value(self, state):
        state.fields.velocity = np.full(
            (state.domain.size, 2),
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.pressure = np.full(
            state.domain.size,
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.density = np.full(
            state.domain.size,
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.phase_field = np.full(
            state.domain.size,
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.solid_id = np.full(
            state.domain.size,
            fill_value=state.domain.mpi_rank,
            dtype=np.int32
        )
        state.fields.solid = np.full(
            state.domain.size,
            fill_value=(state.domain.mpi_rank % 2),
            dtype=np.bool_
        )
        state.fields.solid_boundary = np.full(
            state.domain.size,
            fill_value=(state.domain.mpi_rank % 2),
            dtype=np.bool_
        )
        state.fields.fluid_boundary = np.full(
            state.domain.size,
            fill_value=(state.domain.mpi_rank % 2),
            dtype=np.bool_
        )
        state.fields.pop_fluid = np.full(
            (state.domain.size, 9),
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.pop_phase = np.full(
            (state.domain.size, 9),
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )
        state.fields.grad_phase_field = np.full(
            (state.domain.size, 2),
            fill_value=state.domain.mpi_rank,
            dtype=state.control.precision
        )

    def compare_buffers_edges(
        self,
        state,
        rank,
        bool_buffers,
        int_buffers,
        float_buffers,
        direction="x",
        val=0
    ):
        float_value = state.control.precision(rank)
        int_value = np.int32(rank)
        bool_value = np.bool_(rank % 2)
        if direction == "x":
            for buffer in bool_buffers:
                field = getattr(state.fields, buffer)
                for y in range(1, state.domain.shape[1] - 1):
                    ind = val * state.domain.shape[1] + y
                    if len(field.shape) == 1:
                        assert field[ind] is bool_value
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert field[ind, k] is bool_value
            for buffer in int_buffers:
                field = getattr(state.fields, buffer)
                for y in range(1, state.domain.shape[1] - 1):
                    ind = val * state.domain.shape[1] + y
                    if len(field.shape) == 1:
                        assert field[ind] == int_value
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert field[ind, k] == int_value
            for buffer in float_buffers:
                field = getattr(state.fields, buffer)
                for y in range(1, state.domain.shape[1] - 1):
                    ind = val * state.domain.shape[1] + y
                    if len(field.shape) == 1:
                        assert np.allclose(field[ind], float_value)
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert np.allclose(field[ind, k], float_value)
        elif direction == "y":
            for buffer in bool_buffers:
                field = getattr(state.fields, buffer)
                for x in range(1, state.domain.shape[0] - 1):
                    ind = x * state.domain.shape[1] + val
                    if len(field.shape) == 1:
                        assert field[ind] is bool_value
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert field[ind, k] is bool_value
            for buffer in int_buffers:
                field = getattr(state.fields, buffer)
                for x in range(1, state.domain.shape[0] - 1):
                    ind = x * state.domain.shape[1] + val
                    if len(field.shape) == 1:
                        assert field[ind] == int_value
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert field[ind, k] == int_value
            for buffer in float_buffers:
                field = getattr(state.fields, buffer)
                for x in range(1, state.domain.shape[0] - 1):
                    ind = x * state.domain.shape[1] + val
                    if len(field.shape) == 1:
                        assert np.allclose(field[ind], float_value)
                    elif len(field.shape) == 2:
                        for k in range(field.shape[1]):
                            assert np.allclose(field[ind, k], float_value)

    def compare_buffers_corners(
        self,
        state,
        rank,
        bool_buffers,
        int_buffers,
        float_buffers,
        x=0,
        y=0
    ):
        ind = x * state.domain.shape[1] + y
        float_value = state.control.precision(rank)
        int_value = np.int32(rank)
        bool_value = np.bool_(rank % 2)
        for buffer in bool_buffers:
            field = getattr(state.fields, buffer)
            field[ind] = bool_value
        for buffer in int_buffers:
            field = getattr(state.fields, buffer)
            field[ind] = int_value
        for buffer in float_buffers:
            field = getattr(state.fields, buffer)
            field[ind] = float_value

    def test_exchange_all_periodic(self, decomposition):
        nx, ny = decomposition
        decompose_dict = make_decompose_dict(nx, ny)
        simulation = make_simulation(decompose_dict=decompose_dict)
        state = DummyState(
            comm,
            simulation,
            fluid=True,
            phase=True
        )
        self.set_fields_value(state)
        mpi_operator = MPIOperator(comm, state)
        bool_buffers = ["solid", "solid_boundary", "fluid_boundary"]
        int_buffers = ["solid_id"]
        float_buffers = [
            "velocity",
            "pressure",
            "density",
            "phase_field",
            "grad_phase_field",
            "pop_fluid",
            "pop_phase"
        ]
        mpi_operator.halo_exchange(
            comm,
            state,
            bool_buffers=bool_buffers,
            int_buffers=int_buffers,
            float_buffers=float_buffers
        )

        # Check neighboring ranks
        i_proc = state.domain.i_proc
        j_proc = state.domain.j_proc
        i_proc_left = (i_proc - 1 + nx) % nx
        i_proc_right = (i_proc - 1 + nx) % nx
        j_proc_top = (j_proc + 1 + ny) % ny
        j_proc_bottom = (j_proc - 1 + ny) % ny
        left_rank = i_proc_left * ny + j_proc
        right_rank = i_proc_right * ny + j_proc
        top_rank = i_proc * ny + j_proc_top
        bottom_rank = i_proc * ny + j_proc_bottom
        assert mpi_operator.left_rank == left_rank
        assert mpi_operator.right_rank == right_rank
        assert mpi_operator.top_rank == top_rank
        assert mpi_operator.bottom_rank == bottom_rank

        # Compare edge buffers
        self.compare_buffers_edges(
            state,
            left_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            direction="x",
            val=0
        )
        self.compare_buffers_edges(
            state,
            right_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            direction="x",
            val=(state.domain.shape[0] - 1)
        )
        self.compare_buffers_edges(
            state,
            bottom_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            direction="y",
            val=0
        )
        self.compare_buffers_edges(
            state,
            top_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            direction="y",
            val=(state.domain.shape[1] - 1)
        )

        # Compare corners
        left_top_rank = i_proc_left * ny + j_proc_top
        self.compare_buffers_corners(
            state,
            left_top_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            x=0,
            y=(state.domain.shape[1] - 1)
        )
        right_top_rank = i_proc_right * ny + j_proc_top
        self.compare_buffers_corners(
            state,
            right_top_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            x=(state.domain.shape[0] - 1),
            y=(state.domain.shape[1] - 1)
        )
        left_bottom_rank = i_proc_left * ny + j_proc_bottom
        self.compare_buffers_corners(
            state,
            left_bottom_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            x=0,
            y=0
        )
        right_bottom_rank = i_proc_right * ny + j_proc_bottom
        self.compare_buffers_corners(
            state,
            right_bottom_rank,
            bool_buffers,
            int_buffers,
            float_buffers,
            x=(state.domain.shape[0] - 1),
            y=0
        )
