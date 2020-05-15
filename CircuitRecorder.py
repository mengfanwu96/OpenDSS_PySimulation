import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class DirectRecord():
    def __init__(self, dss, time_steps):
        self.line_currents = {}
        self.bus_voltages = {}

        self.total_step = time_steps

        self.bus_vBase = {}
        self.bus_phase = {}
        for name, bus_obj in dss.bus_class_dict.items():
            self.bus_vBase[name] = bus_obj.kVBase
            self.bus_voltages[name] = np.zeros((3, time_steps), dtype=complex)
            self.bus_phase[name] = np.array(bus_obj.phase_loc)

        for x in dss.line_dict.keys():
            self.line_currents[x] = np.zeros((3, time_steps), dtype=complex)

        self.current_step = None

    def record(self, dss, step):
        for x, handle in dss.line_dict.items():
            current_vec = np.array(dss.circuit.CktElements(handle).currents)
            phase_idx = dss.line_class_dict[x].phase_idx
            cidx = 2 * np.array(range(0, min(len(current_vec) // 4, 3)))
            self.line_currents[x][phase_idx, step] = current_vec[cidx] + 1j * current_vec[cidx + 1]

        for bus, obj in dss.bus_class_dict.items():
            voltage_vec = np.array(dss.circuit.Buses(obj.handle).puVoltages)
            phase_idx = self.bus_phase[bus] - 1    # not compatible with the phase loc above in currents
            cidx = 2 * np.array(range(0, min(len(voltage_vec) // 2, 3)))
            self.bus_voltages[bus][phase_idx, step] = voltage_vec[cidx] + 1j * voltage_vec[cidx + 1]

        self.current_step = step


    def plot_busV(self, bus_name=""):
        if bus_name not in self.bus_voltages.keys():
            print("Bus not found.")
        else:
            phase_num = len(self.bus_phase[bus_name])
            phase_idx = self.bus_phase[bus_name] - 1
            v_amp = np.abs(self.bus_voltages[bus_name][phase_idx, :])

            fig, axes = plt.subplots(phase_num, 1)
            if phase_num == 1:
                axes = [axes]
            fig.suptitle("V/u of Bus %s, base=%.1f" % (bus_name, self.bus_vBase[bus_name]), y=1)

            for idx, ax in enumerate(axes):
                data = v_amp[idx, :]
                stat = self.stat_analysis(data)
                ax.set_ylim(0.87, 1.13)
                ax.plot(data)
                ax.set_title("Phase %s, avg %.2f, std %.2f extreme %s" %
                             (self.bus_phase[bus_name][idx], stat['avg'], stat['std'], stat['ext']))
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

    def stat_analysis(self, data):
        res = {}
        res['avg'] = np.average(data)
        res['std'] = np.std(data)
        extreme_point_1 = data > 1.05
        extreme_point_2 = data < 0.95
        a = Counter(extreme_point_1)[True]
        b = Counter(extreme_point_2)[True]
        res['ext'] = a + b
        return res