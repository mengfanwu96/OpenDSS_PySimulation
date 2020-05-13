import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter


class DirectRecord():
    def __init__(self, dss, time_steps):
        # self.node_names = circuit.AllNodeNames
        # self.line_names = circuit.Lines.AllNames
        # self.node_voltages = np.zeros((len(self.node_names), time_steps))
        self.line_currents = {}
        self.bus_voltages = {}

        self.total_step = time_steps

        self.bus_vBase = {}
        for idx, x in enumerate(dss.bus_list):
            self.bus_vBase[x] = dss.circuit.Buses(idx).kVBase
            self.bus_voltages[x] = np.zeros((3, time_steps), dtype=complex)

        for x in dss.line_dict.keys():
            self.line_currents[x] = np.zeros((3, time_steps), dtype=complex)

        self.bus_phase = dss.bus_phase_dict

    # def allocate2dict(self):
    #     self.node_voltages_dict = {}
    #     for idx, x in enumerate(self.node_names):
    #         bus = x.split(".")[0]
    #         if bus not in self.node_voltages_dict.keys():
    #             self.node_voltages_dict[bus] = {}
    #             self.dictionary_available = True
    #
    #         phase = int(x.split(".")[1])
    #         self.node_voltages_dict[bus][phase] = self.node_voltages[idx, :]

    def record(self, dss, step):
        for x, handle in dss.line_dict.items():
            current_vec = np.array(dss.circuit.CktElements(handle).currents)
            phase_loc = dss.line_class_dict[x].phase_loc
            cidx = 2 * np.array(range(0, min(len(current_vec) // 4, 3)))
            self.line_currents[x][phase_loc, step] = current_vec[cidx] + 1j * current_vec[cidx + 1]

        for idx, x in enumerate(dss.bus_list):
            voltage_vec = np.array(dss.circuit.Buses(idx).puVoltages)
            phase_loc = np.array(dss.bus_phase_dict[x]) - 1
            cidx = 2 * np.array(range(0, min(len(voltage_vec) // 2, 3)))
            self.bus_voltages[x][phase_loc, step] = voltage_vec[cidx] + 1j * voltage_vec[cidx + 1]


    def plot_busV(self, bus_name=""):
        if bus_name not in self.bus_voltages.keys():
            print("Bus not found.")
        else:
            phase_num = len(self.bus_phase[bus_name])
            phase_loc = np.array(self.bus_phase[bus_name]) - 1
            v_amp = np.abs(self.bus_voltages[bus_name][phase_loc, :])

            fig, axes = plt.subplots(phase_num, 1)
            if phase_num == 1:
                axes = [axes]
            fig.suptitle("V/u of Bus %s, base=%.1f" % (bus_name, self.bus_vBase[bus_name]), y=1)

            for idx, ax in enumerate(axes):
                ax.set_ylim(0.85, 1.15)
                ax.plot(v_amp[idx, :])
                ax.set_title("Phase %s" % self.bus_phase[bus_name][idx])
            # plt.show()
            fig.show()

    def plot_lineC(self, line):
        currents = np.abs(self.line_currents[line])
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("current on L%s" % line, y=1)
        for phase_idx in range(3):
            ax[phase_idx].plot(currents[phase_idx, :])
            ax[phase_idx].set_title("Phase %s" % (phase_idx+1))
        fig.show()