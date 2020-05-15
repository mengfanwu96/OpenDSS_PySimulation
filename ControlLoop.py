import numpy as np
from collections import Counter
from CircuitComponents import DSS
from CircuitRecorder import DirectRecord

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


class CapacitorControl:
    def __init__(self, dss):
        self.state_continuity = {}
        self.numsteps = {}
        self.capacitance = {}
        self.capacitance_slot = {}
        self.momentum = 7

        for x, handle in dss.capacitor_dict.items():
            self.state_continuity[x] = self.momentum
            dss.circuit.Capacitors.Name = x
            self.numsteps[x] = dss.circuit.Capacitors.NumSteps
            self.capacitance[x] = dss.circuit.Capacitors.kvar
            self.capacitance_slot[x] = self.capacitance[x] / self.numsteps[x]

        self.capacitor_init(dss.circuit, list(dss.capacitor_dict.keys()))

    def capacitor_init(self, circuit, cap_list, code=None):   # some problem: initial state might be perfect, why here letting half of the capacitance on?
        for x in cap_list:
            if code is None:
                code = 0
            state = [1 for x in range(code)]
            state += [0 for x in range(self.numsteps[x] - code)]
            state = tuple(state)
            circuit.Capacitors.Name = x
            circuit.Capacitors.States = state

    def observe_circuit_voltage(self, dss: DSS, bus, rc: DirectRecord):
        node_voltages_pu = {}
        for name in bus:
            for i in dss.bus_class_dict[name].phase_loc:
                node = name + '.' + str(i)
                node_voltages_pu[node] = np.abs(rc.bus_voltages[name][i-1, rc.current_step])

        return node_voltages_pu

    def observe_node_power(self, dss: DSS, bus, rc: DirectRecord):
        capacitance_aug = {}
        for name in bus:
            for i in dss.bus_class_dict[name].phase_loc:
                node = name + '.' + str(i)
                V_pu = rc.bus_voltages[name][i-1, rc.current_step]
                if np.abs(V_pu) >= 1:
                    capacitance_aug[node] = 0
                else:
                    V =  V_pu * rc.bus_vBase[name]
                    upstream = dss.bus_class_dict[name].upstream
                    line = dss.find_line(upstream, name)
                    assert line is not None, "line not found"
                    I = rc.line_currents[line][i-1, rc.current_step]
                    angle_diff = (np.angle(V) - np.angle(I)) % np.pi
                    kVA = np.abs(V * I)
                    kVA_std = kVA / (np.abs(V_pu) ** 2)
                    capacitance_aug[node] = kVA_std * np.sin(angle_diff)
        return capacitance_aug

    def control_action(self, dss, node_info_dict, mode):
        for cap in self.numsteps.keys():
            if self.state_continuity[cap] < self.momentum:
                self.state_continuity[cap] += 1
                return
            else:
                dss.circuit.setActiveElement('Capacitor.' + cap)
                bus_connected = dss.circuit.ActiveCktElement.Properties("Bus1").Val
                action_code = 0
                if bus_connected in node_info_dict.keys():
                    if mode == 'v':
                        voltage_slot = 0.1 / self.numsteps[cap]
                        action_code = np.round((1 - node_info_dict[bus_connected]) / voltage_slot)
                    elif mode == 'power':
                        action_code = np.round(node_info_dict[bus_connected] / self.capacitance_slot[cap])
                    self.turn_capacitor(dss.circuit, cap, action_code)


    def turn_capacitor(self, circuit, cap, num):
        num = max(num, 0)
        num = min(num, self.numsteps[cap])

        circuit.Capacitors.Name = cap
        oldStates = list(circuit.Capacitors.States)
        c = Counter(oldStates)

        if c[0] == self.numsteps[cap] - num or c[1] == num:
            self.state_continuity[cap] += 1
            return
        else:
            for step in range(len(oldStates)):
                oldStates[step] = 1 if step < num else 0
            self.state_continuity[cap] = 0
            circuit.Capacitors.States = tuple(oldStates)