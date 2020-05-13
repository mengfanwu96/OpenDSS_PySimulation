import numpy as np
from collections import Counter


def observe_circuit_current(dss, line):
    line_threshold = {'670671':[1400, 500, 750], '692675':[1200, 200, 600]}
    current = []
    for x in line:
        current.append(dss.circuit.CktElements(dss.line_dict[x]).currents)

    current = np.array(current)

    cur = np.zeros((2, 3), dtype=complex)
    for idx in range(len(current)):
        cidx = np.array([0, 2, 4])  # not reasonable, later there might be < 3 phase current
        cur[idx, :] = current[idx, cidx] + 1j * current[idx, cidx + 1]

    cur_amp = np.abs(cur)

    res = {}
    for idx, x in enumerate(line):
        ratio = np.divide(cur_amp[idx, :], np.array(line_threshold[x]))

        res[x] = int(np.max(ratio) / 0.25)

    return res


def capacitor_init(circuit, cap_list, code=None):
    circuit.Capacitors.Name = cap_list[0]
    numsteps = int(circuit.Capacitors.NumSteps)

    if code is None:
        code = numsteps // 2
    state = [1 for x in range(code)]
    state += [0 for x in range(numsteps-code)]
    state = tuple(state)
    for x in cap_list:
        circuit.Capacitors.Name = x
        circuit.Capacitors.States = state



def capacitor_controller(dss, line, action, cap_ctrl):
    capacitors = dss.capacitor_dict

    # for line, action in actions.items():
    for cap, handle in capacitors.items():
        dss.circuit.setActiveElement('Capacitor.' + cap)
        if dss.circuit.ActiveCktElement.Properties("Bus1").Val in line:
            cap_ctrl.turn_capacitor(dss.circuit, cap, action)


class capacitor_ctrl:
    def __init__(self, dss):
        self.state_continuity = {}
        self.numsteps = {}

        for x, handle in dss.capacitor_dict.items():
            self.state_continuity[x] = 0
            dss.circuit.Capacitors.Name = x
            self.numsteps[x] = dss.circuit.Capacitors.NumSteps

    def turn_capacitor(self, circuit, cap, num):
        if self.state_continuity[cap] < 5:
            self.state_continuity[cap] += 1
            return
        else:
            num = min(num, 0)
            num = max(num, self.numsteps[cap])

            circuit.Capacitors.Name = cap
            oldStates = list(circuit.Capacitors.States)
            c = Counter(oldStates)

            if c[1] == num:
                self.state_continuity[cap] += 1
                return
            else:
                for step in range(len(oldStates)):
                    oldStates[step] = 1 if step < num else 0
                self.state_continuity[cap] = 0
                circuit.Capacitors.States = tuple(oldStates)

    def observe_circuit_voltage(dss, bus, rc):
        bus_voltages = []
        for name in bus:
            idx = dss.circuit.setActiveBus(name)
            bus_voltages.append(dss.circuit.Buses(idx).puVoltages)

        bus_voltages = np.array(bus_voltages)
        # cplx_bus_voltages = np.zeros((2, 3))

        cidx = np.array([0, 2, 4])  # not reasonable, later there might be < 3 phase
        cplx_bus_voltages = bus_voltages[:, cidx] + 1j * bus_voltages[:, cidx + 1]
        abs_bus_voltages = np.abs(cplx_bus_voltages)

        # tap_control = (abs_bus_voltages - 1) / 0.05 + 2
        tap_control = np.round((1 - abs_bus_voltages) / voltage_slot).astype(int) + numsteps / 2

        bus_code = {}
        for i in range(2):
            for j in range(3):
                bus_code[bus[i] + '.' + str(j + 1)] = tap_control[i, j]
        # return {bus[0]:cap_control[0], bus[1]:cap_control[1]}
        return bus_code

