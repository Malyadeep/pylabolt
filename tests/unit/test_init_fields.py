import numpy as np
import pytest

import pylabolt.base.init_fields as init_fields
from factories import make_control, make_domain, make_fields, make_simulation


""" Fixtures """


@pytest.fixture
def setup_env():
    control = make_control()
    domain = make_domain(5, 5, 0)
    fields = make_fields(domain, control)
    return control, domain, fields


@pytest.fixture
def patched_local_to_global(monkeypatch):
    monkeypatch.setattr(
        init_fields,
        "local_to_global",
        lambda i, j, offset: (i, j)
    )


"""
read_dict tests
"""


def test_read_dict_fixed_scalar(setup_env):
    control, domain, fields = setup_env
    dict_input = {
        "type": "fixed",
        "value": 3
    }
    init_fields.read_dict(
        dict_input,
        fields.density,
        fields.ghost_node,
        domain,
        control,
        scalar_dict=True
    )

    assert np.all(fields.density[~fields.ghost_node] == 3)


def test_read_dict_fixed_vector(setup_env):
    control, domain, fields = setup_env
    dict_input = {
        "type": "fixed",
        "value": [1, 2]
    }
    init_fields.read_dict(
        dict_input,
        fields.velocity,
        fields.ghost_node,
        domain,
        control,
        scalar_dict=False
    )

    assert np.all(fields.velocity[~fields.ghost_node, :] == [1, 2])


def test_read_dict_func_scalar(setup_env, patched_local_to_global):
    control, domain, fields = setup_env

    def dummy_func(i, j):
        return i * j

    dict_input = {
        "type": "func",
        "func": dummy_func
    }
    init_fields.read_dict(
        dict_input,
        fields.density,
        fields.ghost_node,
        domain,
        control,
        scalar_dict=True
    )

    expected_result = np.zeros(domain.size, dtype=control.precision)
    for ind in range(domain.size):
        if not fields.ghost_node[ind]:
            i = ind // domain.shape[1]
            j = ind - i * domain.shape[1]
            expected_result[ind] = (i - 1) * (j - 1)

    assert np.all(fields.density == expected_result)


def test_read_dict_func_vector(setup_env, patched_local_to_global):
    control, domain, fields = setup_env

    def dummy_func(i, j):
        return i, j

    dict_input = {
        "type": "func",
        "func": dummy_func
    }
    init_fields.read_dict(
        dict_input,
        fields.velocity,
        fields.ghost_node,
        domain,
        control,
        scalar_dict=False
    )

    expected_result = np.zeros((domain.size, 2), dtype=control.precision)
    for ind in range(domain.size):
        if not fields.ghost_node[ind]:
            i = ind // domain.shape[1]
            j = ind - i * domain.shape[1]
            expected_result[ind, 0] = i - 1
            expected_result[ind, 1] = j - 1

    assert np.all(fields.velocity == expected_result)


def test_read_dict_missing_type(setup_env):
    control, domain, fields = setup_env

    dict_input = {}

    mssg = "type missing in field definition"
    with pytest.raises(ValueError, match=mssg):
        init_fields.read_dict(
            dict_input,
            fields.velocity,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=False
        )


def test_read_dict_missing_value(setup_env):
    control, domain, fields = setup_env

    dict_input = {
        "type": "fixed"
    }

    mssg = "value missing for fixed type field definition"
    with pytest.raises(ValueError, match=mssg):
        init_fields.read_dict(
            dict_input,
            fields.velocity,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=False
        )


def test_read_dict_missing_func(setup_env):
    control, domain, fields = setup_env

    dict_input = {
        "type": "func"
    }

    mssg = "func missing for func type field definition"
    with pytest.raises(ValueError, match=mssg):
        init_fields.read_dict(
            dict_input,
            fields.velocity,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=False
        )


"""
init_fields tests
"""


def test_init_fields_missing_initial_fields(setup_env):
    control, domain, fields = setup_env

    simulation = make_simulation()

    mssg = "initial_fields_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        init_fields.init_fields(
            simulation,
            control,
            domain,
            fields,
            fluid=True,
            phase=True,
            scalar=False,
            verbose=False
        )


def test_init_fields_missing_default(setup_env):
    control, domain, fields = setup_env

    simulation = make_simulation(initial_fields_dict={})

    mssg = "default missing in initial_fields_dict"
    with pytest.raises(ValueError, match=mssg):
        init_fields.init_fields(
            simulation,
            control,
            domain,
            fields,
            fluid=True,
            phase=True,
            scalar=False,
            verbose=False
        )


def test_init_fields_default(setup_env):
    control, domain, fields = setup_env

    initial_fields_dict = {
        "default": {
            "fluid": {
                "velocity": {
                    "type": "fixed",
                    "value": [1, 2]
                },
                "density": {
                    "type": "fixed",
                    "value": 3
                },
                "pressure": {
                    "type": "fixed",
                    "value": 4
                }
            },
            "phase": {
                "phase_field": {
                    "type": "fixed",
                    "value": 5
                }
            }
        }
    }

    simulation = make_simulation(initial_fields_dict=initial_fields_dict)

    init_fields.init_fields(
        simulation,
        control,
        domain,
        fields,
        fluid=True,
        phase=True,
        scalar=False,
        verbose=False
    )

    assert np.all(fields.velocity[~fields.ghost_node] == [1, 2])
    assert np.all(fields.velocity[fields.ghost_node] == [0, 0])
    assert np.all(fields.density[~fields.ghost_node] == 3)
    assert np.all(fields.density[fields.ghost_node] == 0)
    assert np.all(fields.pressure[~fields.ghost_node] == 4)
    assert np.all(fields.pressure[fields.ghost_node] == 0)
    assert np.all(fields.phase_field[~fields.ghost_node] == 5)
    assert np.all(fields.phase_field[fields.ghost_node] == 0)


def test_init_fields_override(setup_env):
    control, domain, fields = setup_env

    initial_fields_dict = {
        "default": {
            "fluid": {
                "velocity": {
                    "type": "fixed",
                    "value": [1, 2]
                },
                "density": {
                    "type": "fixed",
                    "value": 3
                },
                "pressure": {
                    "type": "fixed",
                    "value": 4
                }
            },
            "phase": {
                "phase_field": {
                    "type": "fixed",
                    "value": 5
                }
            }
        },
        "region1": {
            "fluid": {
                "velocity": {
                    "type": "fixed",
                    "value": [2, 4]
                },
                "density": {
                    "type": "fixed",
                    "value": 9
                },
                "pressure": {
                    "type": "fixed",
                    "value": 16
                }
            },
            "phase": {
                "phase_field": {
                    "type": "fixed",
                    "value": 25
                }
            }
        }
    }

    simulation = make_simulation(initial_fields_dict=initial_fields_dict)

    init_fields.init_fields(
        simulation,
        control,
        domain,
        fields,
        fluid=True,
        phase=True,
        scalar=False,
        verbose=False
    )

    assert np.all(fields.velocity[~fields.ghost_node] == [2, 4])
    assert np.all(fields.velocity[fields.ghost_node] == [0, 0])
    assert np.all(fields.density[~fields.ghost_node] == 9)
    assert np.all(fields.density[fields.ghost_node] == 0)
    assert np.all(fields.pressure[~fields.ghost_node] == 16)
    assert np.all(fields.pressure[fields.ghost_node] == 0)
    assert np.all(fields.phase_field[~fields.ghost_node] == 25)
    assert np.all(fields.phase_field[fields.ghost_node] == 0)
