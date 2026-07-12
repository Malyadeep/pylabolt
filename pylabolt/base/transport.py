from pylabolt.utils.helpers import print_log


class Transport:
    def __init__(
        self,
        simulation,
        control,
        domain,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Container for transport properties information
        Attributes:

        """
        print_log("-" * 80, domain.mpi_rank, verbose)
        print_log("Setting transport properties...\n",
                  domain.mpi_rank, verbose)
        if not hasattr(simulation, "transport_dict"):
            raise ValueError("transport_dict not found in simulation.py file")
        transport_dict = simulation.transport_dict

        if fluid is True and phase is False:
            if "kin_visc" not in transport_dict:
                raise ValueError("kin_visc missing in transport_dict")
            self.kin_visc = transport_dict["kin_visc"]
            if not isinstance(self.kin_visc, (float, int)):
                raise ValueError("kin_visc must be a float/int")
            self.kin_visc = control.precision(self.kin_visc)
            print_log(
                f"{"kinematic viscosity":20s}: {self.kin_visc}",
                domain.mpi_rank, verbose
            )

        if fluid is True and phase is True:
            property_list = [
                "kin_visc_1",
                "kin_visc_2",
                "density_1",
                "density_2",
                "surface_tension"
            ]
            for property_name in property_list:
                if property_name not in transport_dict:
                    raise ValueError(
                        property_name + " missing in transport_dict"
                    )
                temp_value = transport_dict[property_name]
                if not isinstance(temp_value, (float, int)):
                    raise ValueError(property_name + " must be a float/int")
            self.kin_visc_1 = control.precision(transport_dict["kin_visc_1"])
            self.kin_visc_2 = control.precision(transport_dict["kin_visc_2"])
            self.density_1 = control.precision(transport_dict["density_1"])
            self.density_2 = control.precision(transport_dict["density_2"])
            self.surface_tension = control.precision(
                transport_dict["surface_tension"]
            )

            print_log(
                f"{"kinematic viscosity 1":20s}: {self.kin_visc_1}",
                domain.mpi_rank, verbose
            )
            print_log(
                f"{"kinematic viscosity 2":20s}: {self.kin_visc_2}",
                domain.mpi_rank, verbose
            )
            print_log(
                f"{"density 1":20s}: {self.density_1}",
                domain.mpi_rank, verbose
            )
            print_log(
                f"{"density 2":20s}: {self.density_2}",
                domain.mpi_rank, verbose
            )
            print_log(
                f"{"surface tension":20s}: {self.surface_tension}",
                domain.mpi_rank, verbose
            )

        print_log("Setting transport properties done!",
                  domain.mpi_rank, verbose)
        print_log("-" * 80, domain.mpi_rank, verbose)

    def set_backend(
        self,
        backend,
        fluid=False,
        phase=False,
        scalar=False
    ):
        """
        Configure backend attributes for transport object
        Args:

        Returns:

        """
        if backend.backend_type == "gpu":
            self._device_attrs = []
            if fluid is True and phase is False:
                self._device_attrs = ["kin_visc"]
            if fluid is True and phase is True:
                self._device_attrs = [
                    "kin_visc_1",
                    "kin_visc_2",
                    "density_1",
                    "density_2",
                    "surface_tension"
                ]
            for arg_name in self._device_attrs:
                arg_device = backend.allocate_to_device(
                    getattr(self, arg_name)
                )
                setattr(self, arg_name + "_device", arg_device)
