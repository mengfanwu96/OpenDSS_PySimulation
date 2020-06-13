import win32com.client
import re
import os

class DSS:
    def __init__(self, filename):
        self.engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.engine.Start("0")

        self.text = self.engine.Text
        self.text.Command = "clear"
        self.circuit = self.engine.ActiveCircuit
        self.solution = self.circuit.Solution

        print(self.engine.Version)

        if filename != "":
            self.text.Command = "compile [" + filename + "]"

        # Record bus information
        self.bus_class_dict = {}
        for idx, x in enumerate(self.circuit.AllBusNames):
            self.bus_class_dict[x] = Bus(self, x, idx)
        for node in self.circuit.AllNodeNames:
            name_split = node.split('.')
            bus = name_split[0]
            phase = int(name_split[1])
            self.bus_class_dict[bus].add_phase_loc(phase)
        
        elem_list = self.circuit.AllElementNames
        self.transformer_dict = {}
        self.regcontrol_dict = {}
        self.load_dict = {}
        self.capacitor_dict = {}
        self.capcontrol_dict = {}
        self.line_dict = {}
        self.monitor_dict = {}
        self.pv_dict = {}
        self.storage_dict = {}
        name_module_mapping = {
            "Transformer": self.transformer_dict,
            "RegControl": self.regcontrol_dict,
            "Load": self.load_dict,
            "Capacitor": self.capacitor_dict,
            "CapControl": self.capcontrol_dict,
            "Line": self.line_dict,
            'Monitor': self.monitor_dict,
            "PVSystem2": self.pv_dict,
            "Storage2": self.storage_dict,
        }
        for idx, elem in enumerate(elem_list):
            class_name, elem_name = elem.split('.')
            if class_name in name_module_mapping.keys():
                name_module_mapping[class_name][elem_name] = idx

        self.line_class_dict = {}
        for name in self.line_dict.keys():
            self.line_class_dict[name] = Line(self, name)

    def release_com(self):
        del self.solution
        del self.circuit
        del self.text
        del self.engine

    def find_line(self, upstream, downstream):
        for name, obj in self.line_class_dict.items():
            if obj.bus[0] == upstream and obj.bus[1] == downstream:
                return name
        return None

    def find_regulator(self):
        for x, id in self.transformer_dict.items():
            if '1' in x or '2' in x or '3' in x:
                num_phase = int(self.circuit.CktElements(id).NumPhases)
                if num_phase == 1:
                    reg_name = re.split('([123])', x)
                    return reg_name[0]


class Line:
    def __init__(self, dss: DSS, name):
        self.name = name
        buses = dss.circuit.CktElements(dss.line_dict[name]).BusNames
        self.bus = [x.split('.')[0] for x in buses]
        dss.bus_class_dict[self.bus[1]].add_upstream(self.bus[0])

        phase = buses[0].split('.')
        phase.pop(0)
        self.phase_idx = [int(x)-1 for x in phase]
        self.phase_num = dss.circuit.CktElements(dss.line_dict[name]).NumPhases
        if len(self.phase_idx) == 0 and self.phase_num == 3:
            self.phase_idx = [0, 1, 2]

class Bus:
    def __init__(self, dss: DSS, name, handle):
        self.name = name
        self.handle = handle
        self.phase_num = 0
        self.phase_loc = []
        self.upstream = None
        self.kVBase = float(dss.circuit.Buses(handle).kVBase)

    def add_upstream(self, name):
        self.upstream = name

    def add_phase_loc(self, phase):
        self.phase_loc.append(phase)
        self.phase_loc.sort()
        self.phase_num = len(self.phase_loc)