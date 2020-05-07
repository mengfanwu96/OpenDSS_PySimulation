import os


class LoadVariation:
    def __init__(self, load_list, root='Daily_1min_100profiles'):
        self.load_time_series = {}
        self.std_load = {}
        for idx, load_name in enumerate(load_list):
            with open(root + '\load_profile_%s.txt' % (idx + 1)) as reader:
                loads = reader.readlines()
                self.load_time_series[load_name] = [float(x) for x in loads]
        self.steps = len(self.load_time_series[load_list[0]])
        self.num = len(load_list)
        self.load_list = load_list

    def load_circuit(self, circuit, step, load=None, actual=True):
        if step == 0:
            for load_name in self.load_list:
                circuit.Loads.Name = load_name
                self.std_load[load_name] = circuit.Loads.kW

        if load is None:
            for load_name in self.load_list:
                circuit.Loads.Name = load_name
                if actual:
                    circuit.Loads.kW = self.load_time_series[load_name][step]
                else:
                    circuit.Loads.kW = self.std_load[load_name] * self.load_time_series[load_name][step]
        else:
            assert (load in self.load_list.keys())
            circuit.Loads.Name = load
            if actual:
                circuit.Loads.kW = self.load_time_series[load][step]
            else:
                circuit.Loads.kW = self.std_load[load] * self.load_time_series[load][step]

