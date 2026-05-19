import numpy as np
import pytest
import re

from pylabolt.base.control import Control
from factories import make_simulation


"""
Dummy control dict
"""


def get_control_dict():
    return {
        "start_time": 0,
        "end_time": 1000,
        "std_out_interval": 100,
        "save_interval": 10,
        "checkpoint_interval": None,
        "precision": "double"
    }


"""
Control test
"""


def test_missing_control_dict():
    simulation = make_simulation()

    mssg = "control_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Control(
            simulation,
            0,
            verbose=False
        )


def test_missing_entries():
    exclude_list = [
        "start_time",
        "end_time",
        "std_out_interval",
        "save_interval",
        "checkpoint_interval",
        "precision"
    ]

    for exclude in exclude_list:
        mssg = str(exclude) + " missing in control_dict"
        control_dict = get_control_dict()
        control_dict.pop(exclude)
        with pytest.raises(ValueError, match=mssg):
            Control(
                make_simulation(control_dict=control_dict),
                0,
                verbose=False
            )


def test_control_entries():
    key_list = [
        "start_time",
        "end_time",
        "std_out_interval",
        "save_interval",
        "checkpoint_interval"
    ]
    control_dict = get_control_dict()
    simulation = make_simulation(control_dict=control_dict)
    control = Control(
        simulation,
        0,
        verbose=False
    )
    for key in key_list:
        assert (
            getattr(control, key) ==
            simulation.control_dict[key]
        )


def test_control_precision():
    control_dict = get_control_dict()
    simulation = make_simulation(control_dict=control_dict)
    simulation.control_dict["precision"] = "double"
    control = Control(
        simulation,
        0,
        verbose=False
    )
    assert control.precision == np.float64

    simulation.control_dict["precision"] = "single"
    control = Control(
        simulation,
        0,
        verbose=False
    )
    assert control.precision == np.float32

    simulation.control_dict["precision"] = "other"
    mssg = re.escape(
        "unsupported precision specified." +
        "available precision (single, double)"
    )
    with pytest.raises(ValueError, match=mssg):
        Control(
            simulation,
            0,
            verbose=False
        )
