import numpy as np
import pytest
import re

from pylabolt.base.control import Control


"""
Dummy simulation class
"""


class DummySimulation:
    def __init__(self, exclude=None):
        self.control_dict = {
            "start_time": 0,
            "end_time": 1000,
            "std_out_interval": 100,
            "save_interval": 10,
            "checkpoint_interval": None,
            "precision": "double"
        }
        if exclude is not None:
            self.control_dict.pop(exclude)


"""
Control test
"""


def test_missing_control_dict():
    class Simulation:
        pass

    mssg = "control_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Control(
            Simulation(),
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
        with pytest.raises(ValueError, match=mssg):
            Control(
                DummySimulation(exclude=exclude),
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

    simulation = DummySimulation(exclude=None)
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
    simulation = DummySimulation(exclude=None)
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
