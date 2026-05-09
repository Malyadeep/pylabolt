import h5py
import os
import time
import numpy as np


class reader:
    def __init__(self, fieldList=None, particleList=None,
                 boundaryList=None, fieldReadTime="all"):
        self.fields = {}
        self.obstacles = {}
        self.boundaries = {}
        self.fieldList = fieldList
        self.particleList = particleList
        self.boundaryList = boundaryList
        self.fieldReadTime = fieldReadTime

    def fieldDataReader(self, fileName):
        if self.fieldList is not None:
            while os.path.exists(fileName + ".lock"):
                time.sleep(0.05)
            fieldsDataFile = None
            try:
                fieldsDataFile = h5py.File(fileName, "r")
            except FileNotFoundError:
                print("No fields data file found! Exiting..")
                print("Check fileName:", fileName)
                return
            endTime = fieldsDataFile["parameters"].attrs["endTime"]
            saveInterval = fieldsDataFile["parameters"].attrs["saveInterval"]
            lastValidTimeStep_fields = fieldsDataFile["parameters"].\
                attrs["lastValidTimeStep"]
            Nx = fieldsDataFile["parameters"].attrs["Nx"]
            Ny = fieldsDataFile["parameters"].attrs["Ny"]
            self.fields["Nx"] = Nx
            self.fields["Ny"] = Ny
            self.fields["x"] = np.linspace(0, Nx - 1, Nx)
            self.fields["y"] = np.linspace(0, Ny - 1, Ny)
            if lastValidTimeStep_fields == endTime:
                print("Full time range data available")
                lastValidTimeStep_fields = endTime
            else:
                print("Partial time range data available. last time step =",
                      lastValidTimeStep_fields)
            if self.fieldReadTime == "all":
                times = np.linspace(0, lastValidTimeStep_fields,
                                    lastValidTimeStep_fields//saveInterval + 1)
                self.fields["times"] = times
                group = fieldsDataFile["fields"]
                try:
                    for t, timeStep in enumerate(times):
                        self.fields[str(int(timeStep))] = {}
                        for field in self.fieldList:
                            fieldPath = "t_" + str(int(timeStep)) + "/" +\
                                str(field)
                            tempField = group[fieldPath][:]
                            if len(tempField.shape) == 1:
                                self.fields[str(int(timeStep))][field] =\
                                    tempField.reshape(Nx, Ny)
                            elif len(tempField.shape) == 2:
                                self.fields[str(int(timeStep))][field] =\
                                    tempField.reshape(Nx, Ny, 2)
                except KeyError as e:
                    print("Invalid key in fields file: " + str(e))
            else:
                if (isinstance(self.fieldReadTime, int) or
                        isinstance(self.fieldReadTime, float)):
                    if self.fieldReadTime > lastValidTimeStep_fields:
                        print("ERROR! time-step chosen is beyond" +
                              " available data!")
                        return
                    group = fieldsDataFile["fields"]
                    try:
                        timeStep = self.fieldReadTime
                        self.fields[str(int(timeStep))] = {}
                        for field in self.fieldList:
                            fieldPath = "t_" + str(int(timeStep)) + "/" +\
                                str(field)
                            tempField = group[fieldPath][:]
                            if len(tempField.shape) == 1:
                                self.fields[str(int(timeStep))][field] =\
                                    tempField.reshape(Nx, Ny)
                            elif len(tempField.shape) == 2:
                                self.fields[str(int(timeStep))][field] =\
                                    tempField.reshape(Nx, Ny, 2)
                    except KeyError as e:
                        print("Invalid key in fields file: " + str(e))
                else:
                    print("ERROR! Invalid entry for 'fieldReadTime'")
                    print("Available options: ('all', <int>)")
            fieldsDataFile.close()

    def obstacleDataReader(self, fileName):
        if self.particleList is not None or self.boundaryList is not None:
            while os.path.exists(fileName + ".lock"):
                time.sleep(0.05)
            obstacleDataFile = None
            try:
                obstacleDataFile = h5py.File(fileName, "r")
            except FileNotFoundError:
                print("No obstacle data file found! Exiting..")
                print("Check fileName:", fileName)
                return
            endTime = obstacleDataFile["parameters"].attrs["endTime"]
            saveInterval = obstacleDataFile["parameters"].attrs["saveInterval"]
            lastValidTimeStep_obs = obstacleDataFile["parameters"].\
                attrs["lastValidTimeStep"]
            if lastValidTimeStep_obs == endTime:
                print("Full time range data available")
                lastValidTimeStep_obs = endTime
            else:
                print("Partial time range data available. last time step =",
                      lastValidTimeStep_obs)
            try:
                lastIndex = lastValidTimeStep_obs // saveInterval + 1
                times = np.linspace(0, lastValidTimeStep_obs, lastIndex)
                self.obstacles["times"] = times
                particle_group = obstacleDataFile["particles"]
                datasets = ["position", "force", "forceCap", "forceHyd",
                            "velocity", "torque", "torqueCap", "torqueHyd",
                            "angularVelocity", "inclinationAngle"]
                if self.particleList is not None:
                    for particleName in self.particleList:
                        self.obstacles[particleName] = {}
                        for dataset in datasets:
                            self.obstacles[particleName][dataset] =\
                                particle_group[particleName][
                                    dataset][:lastIndex]
                self.boundaries["times"] = times
                boundary_group = obstacleDataFile["boundaries"]
                datasets = ["force", "forceCap", "forceHyd",
                            "torque", "torqueCap", "torqueHyd"]
                if self.boundaryList is not None:
                    for boundaryName in self.boundaryList:
                        self.boundaries[boundaryName] = {}
                        for dataset in datasets:
                            self.boundaries[boundaryName][dataset] =\
                                boundary_group[boundaryName][
                                    dataset][:lastIndex]
            except KeyError as e:
                print("Invalid key in obstacle file: " + str(e))
            obstacleDataFile.close()
