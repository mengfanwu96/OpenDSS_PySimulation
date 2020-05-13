import numpy as np
import re
import win32com.client

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

        self.bus_list = list(self.circuit.AllBusNames)
        elem_list = self.circuit.AllElementNames
        self.transformer_dict = {}
        self.regcontrol_dict = {}
        self.load_dict = {}
        self.capacitor_dict = {}
        self.capcontrol_dict = {}
        self.line_dict = {}
        self.monitor_dict = {}
        name_module_mapping = {
            "Transformer": self.transformer_dict,
            "RegControl": self.regcontrol_dict,
            "Load": self.load_dict,
            "Capacitor": self.capacitor_dict,
            "CapControl": self.capcontrol_dict,
            "Line": self.line_dict,
            'Monitor': self.monitor_dict,
        }
        for idx, elem in enumerate(elem_list):
            class_name, elem_name = elem.split('.')
            if class_name in name_module_mapping.keys():
                name_module_mapping[class_name][elem_name] = idx

        self.line_class_dict = {}
        for name in self.line_dict.keys():
            self.line_class_dict[name] = Line(self, name)

        self.bus_phase_dict = {}
        for x in self.bus_list:
            self.bus_phase_dict[x] = []

        for node in self.circuit.AllNodeNames:
            name_splitted = node.split('.')
            bus = name_splitted[0]
            phase = int(name_splitted[1])
            self.bus_phase_dict[bus].append(phase)

    def add_command(self, message=""):
        self.text.Command = message


class Line:
    def __init__(self, dss, name):
        self.name = name
        buses = dss.circuit.CktElements(dss.line_dict[name]).BusNames
        self.bus = [x.split('.')[0] for x in buses]

        phase = buses[0].split('.')
        phase.pop(0)
        self.phase_loc = tuple([int(x)-1 for x in phase])
        self.numPhase = dss.circuit.CktElements(dss.line_dict[name]).NumPhases
        if len(self.phase_loc) == 0 and self.numPhase == 3:
            self.phase_loc = tuple([0, 1, 2])
