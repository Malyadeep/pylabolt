import numpy as np

from pylabolt.backend.domain import local_to_global
from pylabolt.utils.helpers import print_log


def init_fields(
    simulation,   # user-defined simulation module
    control,
    domain,
    fields,
    fluid=False,  # True if fluid physics is solved
    phase=False,  # True if multiphase physics is solved
    scalar=False,  # True if scalar transport is solved
    verbose=True
):
    """
    Initializes fields based on user input
    Args:
        simulation
    Returns:

    """
    print_log("-" * 80, domain.mpi_rank, verbose)
    print_log("Initializing fields...\n", domain.mpi_rank, verbose)
    print_log("Reading default initial fields section...",
              domain.mpi_rank, verbose)

    if not hasattr(simulation, "initial_fields_dict"):
        raise ValueError(
            "initial_fields_dict not found in simulation.py file"
        )
    initial_fields_dict = simulation.initial_fields_dict

    if "default" not in initial_fields_dict:
        raise ValueError("default missing in initial_fields_dict")
    default_dict = initial_fields_dict["default"]

    # ------- Initialize default fields ------- #
    if fluid is True:
        if "fluid" not in default_dict:
            raise ValueError(
                "fluid missing in initial_fields_dict - default"
            )
        fluid_dict = default_dict["fluid"]
        init_fields_fluid(
            fluid_dict,
            domain,
            fields,
            control,
            default=True,
            verbose=verbose
        )
    if phase is True:
        if "phase" not in default_dict:
            raise ValueError(
                "'phase' missing in initial_fields_dict - default"
            )
        phase_dict = default_dict["phase"]
        init_fields_phase(
            phase_dict,
            domain,
            fields,
            control,
            default=True,
            verbose=verbose
        )
    if scalar is True:
        # --TODO-- no scalar field solver
        pass

    print_log("Reading default initial fields section done!\n",
              domain.mpi_rank, verbose)

    # ------- Initialize user overrides ------- #
    print_log("Reading user-defined initial fields sections...\n",
              domain.mpi_rank, verbose)
    for key_no, key in enumerate(initial_fields_dict.keys()):
        if key == "default":
            continue

        print_log("Region id: " + str(key_no) +
                  " | Region name: " + str(key), domain.mpi_rank, verbose)
        user_dict = initial_fields_dict[key]

        if "fluid" in user_dict:
            fluid_dict = user_dict["fluid"]
            if fluid is False:
                print_log(
                    "WARNING! not a fluid solver. Skipping fluid overrides",
                    domain.mpi_rank, verbose
                )
            else:
                print_log("fluid: override present", domain.mpi_rank, verbose)
                init_fields_fluid(
                    fluid_dict,
                    domain,
                    fields,
                    control,
                    default=False,
                    verbose=verbose
                )
        else:
            print_log("fluid: no override", domain.mpi_rank, verbose)

        if "phase" in user_dict:
            phase_dict = user_dict["phase"]
            if phase is False:
                print_log(
                    "WARNING! not a multiphase solver." +
                    " Skipping phase overrides", domain.mpi_rank, verbose
                )
            else:
                print_log("phase: override present", domain.mpi_rank, verbose)
                init_fields_phase(
                    phase_dict,
                    domain,
                    fields,
                    control,
                    default=False,
                    verbose=verbose
                )
        else:
            print_log("phase: no override", domain.mpi_rank, verbose)
        print_log("", domain.mpi_rank, verbose)
    print_log("Reading user-defined initial fields section done!\n",
              domain.mpi_rank, verbose)
    print_log("Initializing fields done!", domain.mpi_rank, verbose)
    print_log("-" * 80, domain.mpi_rank, verbose)


def init_fields_fluid(
    fluid_dict,
    domain,
    fields,
    control,
    default=True,
    verbose=True
):
    """
    Args:

    Returns:

    """
    if default is True:
        try:
            velocity_dict = fluid_dict["velocity"]
            density_dict = fluid_dict["density"]
            pressure_dict = fluid_dict["pressure"]
        except KeyError as e:
            raise ValueError("" + str(e) + " is missing in default")
        read_dict(
            velocity_dict,
            fields.velocity,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=False
        )
        read_dict(
            density_dict,
            fields.density,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=True
        )
        read_dict(
            pressure_dict,
            fields.pressure,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=True
        )
    else:
        if "velocity" in fluid_dict:
            print_log(
                "fluid: setting velocity override", domain.mpi_rank, verbose
            )
            read_dict(
                fluid_dict["velocity"],
                fields.velocity,
                fields.ghost_node,
                domain,
                control,
                scalar_dict=False
            )
        else:
            print_log("fluid: no velocity override", domain.mpi_rank, verbose)

        if "density" in fluid_dict:
            print_log(
                "fluid: setting density override", domain.mpi_rank, verbose
            )
            read_dict(
                fluid_dict["density"],
                fields.density,
                fields.ghost_node,
                domain,
                control,
                scalar_dict=True
            )
        else:
            print_log("fluid: no density override", domain.mpi_rank, verbose)

        if "pressure" in fluid_dict:
            print_log(
                "fluid: setting pressure override", domain.mpi_rank, verbose
            )
            read_dict(
                fluid_dict["pressure"],
                fields.pressure,
                fields.ghost_node,
                domain,
                control,
                scalar_dict=True
            )
        else:
            print_log("fluid: no pressure override", domain.mpi_rank, verbose)


def init_fields_phase(
    phase_dict,
    domain,
    fields,
    control,
    default=True,
    verbose=True
):
    """
    Args:

    Returns:

    """
    if default is True:
        try:
            phase_field_dict = phase_dict["phase_field"]
        except KeyError as e:
            print_log("" + str(e) + " is missing in default",
                      domain.mpi_rank, verbose)
        read_dict(
            phase_field_dict,
            fields.phase_field,
            fields.ghost_node,
            domain,
            control,
            scalar_dict=True
        )
    else:
        if "phase_field" in phase_dict:
            print_log("phase: setting phase_field override",
                      domain.mpi_rank, verbose)
            read_dict(
                phase_dict["phase_field"],
                fields.phase_field,
                fields.ghost_node,
                domain,
                control,
                scalar_dict=True
            )
        else:
            print_log(
                "phase: no phase_field override", domain.mpi_rank, verbose
            )


def init_fields_scalar():
    pass


def read_dict(
    dict_input,
    field,
    ghost_node,
    domain,
    control,
    scalar_dict=True
):
    value = None
    func = None
    input_file = None  # TODO: initialization from file
    if "type" not in dict_input:
        raise ValueError("type missing in field definition")
    dict_type = dict_input["type"]

    if dict_type == "fixed":
        if "value" not in dict_input:
            raise ValueError("value missing for fixed type field definition")
        value = dict_input["value"]
        if not scalar_dict and type(value) is list and len(value) == 2:
            value = np.array(value, dtype=control.precision)
        elif scalar_dict and type(value) is float or type(value) is int:
            value = control.precision(value)
        else:
            raise ValueError(
                "vector value must be a list (ux, uy)" +
                " and scalar value must be a float or int"
            )

    elif dict_type == "func":
        if "func" not in dict_input:
            raise ValueError("func missing for func type field definition")
        func = dict_input["func"]
        # TODO: implement some checks to see if function is valid
    else:
        # TODO: Input file type implementation
        raise ValueError("Unsupported velocity initialization")
    if scalar_dict:
        set_field_func = set_field_scalar
    else:
        set_field_func = set_field_vector
    set_field_func(
        domain,
        field,
        ghost_node,
        value=value,
        func=func,
        input_file=input_file
    )


def set_field_vector(
    domain,
    vector_field,
    ghost_node,
    value=None,
    func=None,
    input_file=None
):
    if value is not None:
        vector_field[~ghost_node, :] = value
    elif func is not None:
        for ind in range(domain.size):
            if not ghost_node[ind]:
                i = ind // domain.shape[1]
                j = ind - i * domain.shape[1]
                i_local = i - 1
                j_local = j - 1
                i_global, j_global = local_to_global(
                    i_local, j_local, domain.offset
                )
                vector_field[ind, 0], vector_field[ind, 1] =\
                    func(i_global, j_global)
    elif input_file is not None:
        # TODO: not yet supported
        pass


def set_field_scalar(
    domain,
    scalar_field,
    ghost_node,
    value=None,
    func=None,
    input_file=None
):
    if value is not None:
        scalar_field[~ghost_node] = value
    elif func is not None:
        for ind in range(domain.size):
            if not ghost_node[ind]:
                i = ind // domain.shape[1]
                j = ind - i * domain.shape[1]
                i_local = i - 1
                j_local = j - 1
                i_global, j_global = local_to_global(
                    i_local, j_local, domain.offset
                )
                scalar_field[ind] = func(i_global, j_global)
    elif input_file is not None:
        # TODO: not yet supported
        pass
