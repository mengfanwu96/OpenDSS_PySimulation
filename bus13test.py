import win32com.client
import numpy as np
from collections import Counter
from LoadProfile import LoadVariation
from CircuitComponents import Bus, Branch
from CircuitRecorder import CircuitRecorder
import os
import pickle


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


if __name__ == '__main__':
    overwrite = False
    if overwrite:
        d = DSS(os.getcwd() + "\IEEE13Nodeckt.dss")
        load_list = ['671', '634a', '634b', '634c', '645', '646', '692',
                     '675a', '675b', '675c', '611', '652', '670a', '670b', '670c']

        loading = LoadVariation(load_list)
        d.text.Command = "set mode=daily"
        d.text.Command = "set stepsize=1m"
        # d.solution.Solve()
        rc = CircuitRecorder(d.circuit, 1440)

        for step in range(1440):
            loading.load_circuit(d.circuit, step, actual=False)
            d.solution.Solve()
            rc.record(d.circuit, step)

        with open("circuit_record.pkl", "wb") as output:
            pickle.dump(rc, output, pickle.HIGHEST_PROTOCOL)

    else:
        with open("circuit_record.pkl", "rb") as read:
            rc = pickle.load(read)

    # rc.plot(bus_indexes=[x for x in range(1, 3)])
    rc.plot(branch_indexes=[1])
