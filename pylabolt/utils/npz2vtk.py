import numpy as np
import vtk
import json
import os
from types import SimpleNamespace
from pathlib import Path
import re

from pylabolt.utils.helpers import print_log

from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid


class VTKOperator:
    def __init__(
        self,
        metadata,
        verbose=True
    ):
        """
        Converts raw output to VTK format
        Attributes:

        """
        self.metadata = metadata
        self.validate_metadata(verbose=verbose)
        self.raw_data_path = "output/"
        self.vtk_save_path = "vtk/"

    def validate_metadata(
        self,
        verbose=True
    ):
        """
        Read and validate simulation metadata
        Args:

        Returns:

        """
        try:
            self.mesh = SimpleNamespace(
                size=self.metadata["mesh"]["size"],
                shape=np.array(self.metadata["mesh"]["shape"], dtype=int)
            )
            self.fields_metadata = self.metadata["fields_saved"]
            self.fields = {}
            print_log("\nFields being converted to VTK:", 0, verbose)
            for field_name in self.fields_metadata:
                components = self.fields_metadata[field_name]["components"]
                dtype = self.fields_metadata[field_name]["dtype"]
                if components == 1:
                    field = np.zeros(self.mesh.size, dtype=dtype)
                else:
                    field = np.zeros(
                        (self.mesh.size, components),
                        dtype=dtype
                    )
                self.fields[field_name] = field
                print_log(
                    f"{field_name:<10}: {str(field.dtype):>5}", 0, verbose
                )
            print_log("\n", 0, verbose)
        except KeyError as e:
            raise KeyError(
                "Invalid metadata.json! missing key: " + str(e),
                0, verbose
            )

    def convert_time(
        self,
        time_step,
        verbose=True
    ):
        """
        Convert single time step raw data to VTK
        Args:

        Returns:

        """
        file_path = self.raw_data_path + "t_" + str(time_step) + ".npz"
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                "output file not found for time step: " + str(time_step)
            )
        fields_time_step = np.load(file_path)
        try:
            for field_name in self.fields_metadata:
                metadata_shape = self.fields[field_name].shape
                raw_data_shape = fields_time_step[field_name].shape
                if (raw_data_shape != metadata_shape):
                    raise ValueError(
                        "Size and shape of raw data does not match" +
                        " with metadata.json\n" +
                        "field: " + field_name +
                        " | raw data shape: " + str(raw_data_shape) +
                        " | metadata shape: " + str(metadata_shape)
                    )
                metadata_dtype = str(self.fields[field_name].dtype)
                raw_data_dtype = str(fields_time_step[field_name].dtype)
                if (raw_data_dtype != metadata_dtype):
                    raise ValueError(
                        "dtype of raw data does not match" +
                        " with metadata.json\n" +
                        "field: " + field_name +
                        " | raw data shape: " + raw_data_dtype +
                        " | metadata shape: " + metadata_dtype
                    )
                self.fields[field_name] = fields_time_step[field_name]
        except KeyError as e:
            raise KeyError(
                f"{'missing field':<10}: {str(e):<20}"
                f"{'time':<10}: {time_step:<20}"
            )

        self.write_vtk_fields(time_step, verbose=verbose)

    def write_vtk_fields(
        self,
        time_step,
        verbose=True
    ):
        """
        Write VTK fields to disk
        Args:

        Returns:

        """
        print_log(40 * "-", 0, verbose)
        print_log("Time: " + str(time_step), 0, verbose)
        if not os.path.isdir("vtk"):
            os.makedirs("vtk")
        fields_vtk = {}
        for field_name in self.fields_metadata:
            field_dtype = self.fields_metadata[field_name]["dtype"]
            components = self.fields_metadata[field_name]["components"]
            if field_dtype == "float64" or field_dtype == "float32":
                fields_vtk[field_name] = vtkDoubleArray()
            if field_dtype == "int64" or field_dtype == "bool":
                fields_vtk[field_name] = vtkIntArray()
            fields_vtk[field_name].SetName(field_name)
            if components > 1:
                fields_vtk[field_name].SetNumberOfComponents(components)

        # Create grid
        grid = vtkRectilinearGrid()
        grid.SetDimensions(self.mesh.shape[0], self.mesh.shape[1], 1)

        # Write points and coordinates
        points = vtkIntArray()
        points.SetName('point_ID')
        x = vtkDoubleArray()
        y = vtkDoubleArray()
        z = vtkDoubleArray()

        for i in range(self.mesh.shape[0]):
            x.InsertNextValue(i)

        for j in range(self.mesh.shape[1]):
            y.InsertNextValue(j)

        z = vtkDoubleArray()
        z.InsertNextValue(0.0)

        for j in range(self.mesh.shape[1]):
            for i in range(self.mesh.shape[0]):
                ind = i * self.mesh.shape[1] + j
                points.InsertNextValue(ind)
        grid.SetXCoordinates(x)
        grid.SetYCoordinates(y)
        grid.SetZCoordinates(z)
        grid.GetPointData().AddArray(points)

        # Write Fields
        for field_name in self.fields_metadata:
            print_log("Converting field: " + field_name, 0, verbose)
            for j in range(self.mesh.shape[1]):
                for i in range(self.mesh.shape[0]):
                    ind = i * self.mesh.shape[1] + j
                    field_dtype = self.fields_metadata[field_name]["dtype"]
                    components = self.fields_metadata[field_name]["components"]
                    if components > 1:
                        fields_vtk[field_name].InsertNextTuple(
                            self.fields[field_name][ind]
                        )
                    else:
                        if field_dtype == "bool":
                            fields_vtk[field_name].InsertNextValue(
                                int(self.fields[field_name][ind])
                            )
                        else:
                            fields_vtk[field_name].InsertNextValue(
                                self.fields[field_name][ind]
                            )
            grid.GetPointData().AddArray(fields_vtk[field_name])

        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileVersion(42)
        writer.SetFileName(self.vtk_save_path + "t_" + str(time_step) + ".vtk")
        writer.SetInputData(grid)
        writer.Write()

        print_log("There are " + str(grid.GetNumberOfPoints()) + " points.",
                  0, verbose)
        print_log("There are " + str(grid.GetNumberOfCells()) + " cells.",
                  0, verbose)
        print_log(
            f"{'VTK conversion done, time':<25}:"
            f"{time_step:>5}", 0, verbose
        )
        print_log(40 * "-" + "\n", 0, verbose)

    def convert_multi_time(
        self,
        all=True
    ):
        """
        Convert multiple time step raw data to VTK
        Args:

        Returns:

        """
        if all:
            folder = Path(self.raw_data_path)
            pattern = re.compile(r"^output/t_(\d+)\.npz$")
            save_times = []
            for file_item in folder.iterdir():
                matched_file = pattern.match(str(file_item))
                if matched_file:
                    save_times.append(int(matched_file.group(1)))
            save_times.sort()
            save_times = np.array(save_times, dtype=int)
        for time_step in save_times:
            self.convert_time(time_step)


def convert_to_vtk(
    option,
    time_step=0,
    verbose=True
):
    """
    Converts data to VTK format for visualization in ParaView
    Args:

    Returns:

    """
    print_log("-" * 80, 0, verbose)
    print_log("Converting output to VTK...\n", 0, verbose)
    supported_options = ["all", "time"]
    if option not in supported_options:
        raise ValueError(
            "Invalid VTK conversion option!\n" +
            "Available options: " + str(supported_options)
        )
    print_log(f"{'VTK conversion option':<25}: {option:<5}", 0, verbose)
    if option == "time":
        if not isinstance(time_step, int):
            raise ValueError(
                "VTK conversion time must be an int!"
            )
        print_log(f"{'VTK conversion time':<25}: {time_step:<5}",
                  0, verbose)

    if not os.path.isfile("metadata.json"):
        raise FileNotFoundError(
            "metadata.json file not found in working directory"
        )
    with open("metadata.json") as metadata_file:
        print_log("Reading simulation metadata...", 0, verbose)
        metadata = json.load(metadata_file)
    if not os.path.isdir("output"):
        raise FileNotFoundError(
            "output/ directory not present in working directory"
        )

    vtk_operator = VTKOperator(
        metadata,
        verbose=verbose
    )

    if option == "time":
        vtk_operator.convert_time(time_step)
    elif option == "all":
        vtk_operator.convert_multi_time(all=True)
