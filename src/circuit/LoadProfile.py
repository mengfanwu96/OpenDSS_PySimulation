import os
import re
import matplotlib.pyplot as plt
import numpy as np


class LoadVariation:
    def __init__(self, load_list, root: str, profile_path='/profiles/Daily_1min_100profiles'):
        self.load_time_series = {}
        self.std_load = {}

        abs_profile_path = root + profile_path

        for idx, load_name in enumerate(load_list):
            with open(abs_profile_path + '/load_profile_%s.txt' % (idx + 1)) as reader:
                loads = reader.readlines()
                self.load_time_series[load_name] = np.array([float(x) for x in loads])
        self.load_list = list(load_list)
        self.steps = len(self.load_time_series[self.load_list[0]])
        self.actual = None

    def load_circuit(self, circuit, step, load=None, actual=True):
        self.actual = actual
        if step == 0 and len(self.std_load) == 0:
            for load_name in self.load_list:
                circuit.Loads.Name = load_name
                self.std_load[load_name] = circuit.Loads.kW
                if not actual:
                    self.load_time_series[load_name] = self.load_time_series[load_name] * self.std_load[load_name]
        if load is None:
            for load_name in self.load_list:
                circuit.Loads.Name = load_name
                circuit.Loads.kW = self.load_time_series[load_name][step]
        else:
            assert (load in self.load_list)
            circuit.Loads.Name = load
            circuit.Loads.kW = self.load_time_series[load][step]

    def plot_load_onBus(self, bus, circuit):
        if self.actual is None:
            print("Uncertain about loading power ratio or actual power.")
            return

        relevant_load = []
        for x in self.load_list:
            circuit.setActiveElement('Load.' + x)
            bus_connected = circuit.ActiveCktElement.Properties('bus1').Val
            if bus in bus_connected:
                relevant_load.append(x)

        if len(relevant_load) == 1:
            plt.figure()
            plt.plot(self.load_time_series[relevant_load[0]])
            plt.title("load %s, base power %s" % (relevant_load[0], self.std_load[relevant_load[0]]))
            plt.show()
        else:
            fig, ax = plt.subplots(len(relevant_load) + 1, 1)

            load = np.array([np.array(self.load_time_series[x]) for x in relevant_load])
            load_sum = np.sum(load, axis=0)
            base_power_sum = 0

            for i, load_name in enumerate(relevant_load):
                ax[i].plot(self.load_time_series[load_name])
                ax[i].set_title("load %s, base power %s" % (load_name, self.std_load[load_name]))
                base_power_sum += self.std_load[load_name]
            ax[-1].plot(load_sum)
            ax[-1].set_title("summed load on bus %s, power %s" % (bus, base_power_sum))

            fig.show()