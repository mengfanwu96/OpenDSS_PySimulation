from .CircuitComponents import DSS
import numpy as np


class storage_monitor:
    def __init__(self, dss:DSS, times_steps):
        self.sto = dss.storage_dict
        self.state_dict = {}
        self.stored_dict = {}

        for x in self.sto.keys():
            self.state_dict[x] = np.zeros(times_steps, dtype=str)
            self.stored_dict[x] = np.zeros(times_steps, dtype=float)

    def record(self, circuit, step):
        for x in self.sto.keys():
            circuit.SetActiveElement('Storage2.' + x)
            self.stored_dict[x][step] = circuit.ActiveCktElement.Properties('kWhstored').Val
            self.state_dict[x][step] = circuit.ActiveCktElement.Properties('state').Val
