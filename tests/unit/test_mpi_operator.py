# flake8: noqa: E402

import pytest
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
MPI.Init()
comm = MPI.COMM_WORLD
import re


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
    make_comm,
    make_mesh,
    make_control,
    make_decompose_dict,
    make_fields,
    make_lattice,
    make_simulation,
    make_boundary
)


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
        state.fields.velocity[:, 0] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.velocity[:, 1] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.pressure = np.random.uniform(-1, 1, state.domain.size)
        state.fields.density[:] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.phase_field[:] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.solid[:] = np.bool(np.random.randint(0, 2, state.domain.size))
        state.fields.solid_boundary[:] = np.bool(np.random.randint(0, 2, state.domain.size))
        state.fields.fluid_boundary[:] = np.bool(np.random.randint(0, 2, state.domain.size))
        for k in range(state.lattice.no_of_directions):
            state.fields.pop_fluid[:, k] = np.random.uniform(-1, 1, state.domain.size)
            state.fields.pop_phase[:, k] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.grad_phase_field[:, 0] = np.random.uniform(-1, 1, state.domain.size)
        state.fields.grad_phase_field[:, 1] = np.random.uniform(-1, 1, state.domain.size)
        
    def compare_buffers_y(self, buffer_object, state, x_val=1):
        test_buffer = np.copy(buffer_object.send_buff_left_right)
        for key in list(buffer_object.field_dict.keys()):
            field_to_copy = getattr(state.fields, key)
            field_back_copy =  np.copy(field_to_copy)
            no_of_components = buffer_object.field_dict[key]
            layout_start, layout_end = buffer_object.layout[key]
            send_copy_func = send_copy_y_scalar
            recv_copy_func = recv_copy_y_scalar
            if no_of_components > 1:
                send_copy_func = send_copy_y_vector
                recv_copy_func = recv_copy_y_vector   
            send_copy_func(
                buffer_object.send_buff_left_right,
                field_to_copy,
                layout_start,
                layout_end,
                state.domain.shape,
                x=x_val
            )
            x = x_val
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
            
            recv_copy_func(
                buffer_object.recv_buff_left_right,
                field_back_copy,
                layout_start,
                layout_end,
                state.domain.shape,
                x=x_val
            )
            assert np.all(field_to_copy == field_back_copy)
        assert np.all(buffer_object.send_buff_left_right == test_buffer)

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
        self.compare_buffers_y(
            mpi_operator.bool_buffer,
            state,
            x_val=1
        )
        self.compare_buffers_y(
            mpi_operator.int_buffer,
            state,
            x_val=1
        )
        self.compare_buffers_y(
            mpi_operator.float_buffer,
            state,
            x_val=1
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
        self.compare_buffers_y(
            mpi_operator.bool_buffer,
            state,
            x_val=(state.domain.shape[0] - 2)
        )
        self.compare_buffers_y(
            mpi_operator.int_buffer,
            state,
            x_val=(state.domain.shape[0] - 2)
        )
        self.compare_buffers_y(
            mpi_operator.float_buffer,
            state,
            x_val=(state.domain.shape[0] - 2)
        )
            

# MPI.Finalize()
