import numpy as np
from CircuitComponents import Bus, Branch
import matplotlib.pyplot as plt


class CircuitRecorder:
    def __init__(self, circuit, timestep):
        self.bus = Bus(circuit)
        self.branch = Branch(circuit)

        self.busVoltage = np.zeros((self.bus.V.shape + tuple([timestep, 2])))
        self.branchVoltage = np.zeros((self.branch.V.shape + tuple([timestep, 2])))
        self.branchVoltage2 = np.zeros((self.branch.Vto.shape + tuple([timestep, 2])))
        self.branchCurrent = np.zeros((self.branch.I.shape + tuple([timestep, 2])))

    def record(self, circuit, step):
        self.bus = Bus(circuit)
        self.branch = Branch(circuit)

        for par, data in zip([self.busVoltage, self.branchVoltage, self.branchVoltage2, self.branchCurrent],
                             [self.bus.V, self.branch.V, self.branch.Vto, self.branch.I]):
            par[:, :, step, 0] = np.abs(data)
            par[:, :, step, 1] = np.angle(data) / np.pi * 180

    def plot_item(self, item, idx, message=''):
        plt.figure()
        limits = []

        limits.append(np.min([np.min(item[idx-1, phase, :, 0]) for phase in range(3)]))
        limits.append(np.max([np.max(item[idx-1, phase, :, 0]) for phase in range(3)]))

        limit_range = limits[1] - limits[0]
        limits[0] -= limit_range * 0.1
        limits[1] += limit_range * 0.1

        fig, ax = plt.subplots(3, 2)
        fig.suptitle(message, fontsize=14)
        for phase in range(3):
            for j in range(2):
                if j == 0:
                    ax[phase, j].set_ylim(limits[0], limits[1])
                ax[phase, j].plot(item[idx-1, phase, :, j])
        ax[0, 0].set_title("abs_value", fontsize=8)
        ax[0, 1].set_title("angle in degree", fontsize=8)
        fig.show()

    def plot(self, bus_indexes=None, branch_indexes=None):
        if bus_indexes is not None:
            for bus in bus_indexes:
                if bus <= self.bus.name.shape[0]:
                    self.plot_item(self.busVoltage, bus, message="bus %s voltage" % self.bus.name[bus-1])
                else:
                    print("Bus index overflow")
        if branch_indexes is not None:
            for branch in branch_indexes:
                if branch <= self.branch.name.shape[0]:
                    self.plot_item(self.branchVoltage, branch, message="branch %s voltage1" % self.branch.name[branch-1])
                    self.plot_item(self.branchVoltage2, branch, message="branch %s voltage2" % self.branch.name[branch-1])
                    self.plot_item(self.branchCurrent, branch, message="branch %s current" % self.branch.name[branch-1])
                else:
                    print("Branch index overflow")
