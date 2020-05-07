import win32com.client
import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ColorConverter
import matplotlib.text as text
import re
from collections import Counter
from LoadProfile import LoadVariation
from CircuitComponents import Bus, Branch
import os

colorConverter = ColorConverter()


class DSS:
    def __init__(self, filename=""):
        self.engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.engine.Start("0")

# use the Text interface to OpenDSS
        self.text = self.engine.Text
        self.text.Command = "clear"
        self.circuit = self.engine.ActiveCircuit
        self.solution = self.circuit.Solution
        self.elem = self.circuit.ActiveCktElement

        print(self.engine.Version)

        if filename != "":
            self.text.Command = "compile [" + filename + "]"

    def populate_results(self):
        self.bus = Bus(self.circuit)
        self.branch = Branch(self.circuit)

    def add_command(self, message=""):
        self.text.Command = message


class CircuitRecord:
    def __init__(self, circuit, timestep):
        self.bus = Bus(circuit)
        self.branch = Branch(circuit)

        self.busVoltage = np.zeros((self.bus.V.shape + tuple([timestep])), dtype=np.complex128)
        self.branchVoltage = np.zeros((self.branch.V.shape + tuple([timestep])), dtype=np.complex128)
        self.branchVoltage2 = np.zeros((self.branch.Vto.shape + tuple([timestep])), dtype=np.complex128)
        self.branchCurrent = np.zeros((self.branch.I.shape + tuple([timestep])), dtype=np.complex128)

        self.busVoltage[:, :, 0] = self.bus.V
        self.branchVoltage[:, :, 0] = self.branch.V
        self.branchVoltage2[:, :, 0] = self.branch.Vto
        self.branchCurrent[:, :, 0] = self.branch.I

    def record(self, circuit, step):
        self.bus = Bus(circuit)
        self.branch = Branch(circuit)
        self.busVoltage[:, :, step] = self.bus.V
        self.branchVoltage[:, :, step] = self.branch.V
        self.branchVoltage2[:, :, step] = self.branch.Vto
        self.branchCurrent[:, :, step] = self.branch.I

    def plot(self, bus_index, branch_index):
        pass


if __name__ == '__main__':
    d = DSS(os.getcwd() + "\IEEE13Nodeckt.dss")
    load_list = ['671', '634a', '634b', '634c', '645', '646', '692',
                 '675a', '675b', '675c', '611', '652', '670a', '670b', '670c']

    loading = LoadVariation(load_list)
    loading.load_circuit(d.circuit, 0, actual=False)
    d.text.Command = "set mode=daily"
    d.text.Command = "set stepsize=1m"
    d.solution.Solve()
    rc = CircuitRecord(d.circuit, 1440)

    for i in range(1, 1440):
        loading.load_circuit(d.circuit, i, actual=False)
        d.solution.Solve()
        rc.record(d.circuit, i)

    branch_rec = rc.branchVoltage[0,0,:]
    c = Counter(np.abs(branch_rec))
    rc.plot()