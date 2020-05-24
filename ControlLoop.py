import numpy as np
from collections import Counter
from CircuitComponents import DSS
from CircuitRecorder import DirectRecord
import matplotlib.pyplot as plt


class CapacitorControl:
    def __init__(self, dss):
        self.state_continuity = {}
        self.numsteps = {}
        self.capacitance = {}
        self.capacitance_slot = {}
        self.cap_to_controlBus = {}     # key: capacitor, value: bus.phase
        self.cap_control_log = {}
        self.cap_last_update = {}
        self.init_state = None
        self.momentum = 5

        for x, handle in dss.capacitor_dict.items():
            self.state_continuity[x] = self.momentum
            dss.circuit.Capacitors.Name = x
            self.numsteps[x] = dss.circuit.Capacitors.NumSteps
            self.capacitance[x] = dss.circuit.Capacitors.kvar
            self.capacitance_slot[x] = self.capacitance[x] / self.numsteps[x]

        self.capacitor_init(dss.circuit, list(dss.capacitor_dict.keys()))

    def capacitor_init(self, circuit, cap_list, code=None):   # some problem: initial state can not be specified per capacitor
        for x in cap_list:
            if code is None:
                code = 0
                self.init_state = 0
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
                V = V_pu * rc.bus_vBase[name]
                upstream = dss.bus_class_dict[name].upstream
                line = dss.find_line(upstream, name)
                assert line is not None, "line not found"
                I = rc.line_currents[line][i-1, rc.current_step]
                angle_diff = np.angle(V) - np.angle(I)
                kVA = np.abs(V * I)
                kVA_std = kVA / (np.abs(V_pu) ** 2)
                capacitance_aug[node] = kVA_std * np.sin(angle_diff)

        if rc.current_step == 0:
            for cap in self.numsteps.keys():
                dss.circuit.SetActiveElement('Capacitor.' + cap)
                bus_connected = dss.circuit.ActiveCktElement.Properties("Bus1").Val
                if bus_connected in capacitance_aug.keys():
                    self.cap_to_controlBus[cap] = bus_connected
                    self.cap_control_log[cap] = np.zeros(rc.total_step)
                    self.cap_control_log[cap][0] = self.init_state
                    self.cap_last_update[cap] = 0

        return capacitance_aug

    def control_action(self, dss, node_info_dict, time, mode):
        for cap, bus_connected in self.cap_to_controlBus.items():
            if self.cap_last_update[cap] != 0:
                assert self.cap_last_update[cap] + self.state_continuity[cap] == time, \
                    'time %s, last update %s, span %s' % (time, self.cap_last_update[cap], self.state_continuity[cap])
            if self.state_continuity[cap] <= self.momentum:
                self.state_continuity[cap] = self.state_continuity[cap] + 1
                continue
            else:
                action_code = 0
                if mode == 'v':
                    voltage_slot = 0.1 / self.numsteps[cap]
                    action_code = np.round((1 - node_info_dict[bus_connected]) / voltage_slot)
                elif mode == 'power':
                    action_code = np.round(node_info_dict[bus_connected] / self.capacitance_slot[cap])
                # self.turn_capacitor(dss.circuit, cap, action_code * np.exp(action_code * (np.log(2) / self.numsteps[cap])))
                self.turn_capacitor(dss.circuit, cap, action_code, time)

    def turn_capacitor(self, circuit, cap, num, time):
        circuit.Capacitors.Name = cap
        oldStates = list(circuit.Capacitors.States)
        c = Counter(oldStates)
        if 1 in c.keys():
            old_state = c[1]
        else:
            old_state = 0

        num += old_state
        num = max(num, 0)
        num = min(num, self.numsteps[cap])

        if self.cap_last_update[cap] != 0:
            assert self.cap_last_update[cap] + self.state_continuity[cap] == time, \
                'time %s, last update %s, span %s' % (time, self.cap_last_update[cap], self.state_continuity[cap])
            # TODO: cancel this assertion, only use time_span

        if c[0] == self.numsteps[cap] - num or c[1] == num:     # if the new action code does not change the current state
            self.state_continuity[cap] = self.state_continuity[cap] + 1
            return
        else:
            self.cap_control_recorder(cap, time, num)
            self.cap_last_update[cap] = time
            for step in range(len(oldStates)):
                oldStates[step] = 1 if step < num else 0
            self.state_continuity[cap] = 1
            circuit.Capacitors.States = tuple(oldStates)

    def cap_control_recorder(self, cap, time, num):
        if time < len(self.cap_control_log[cap]):
            self.cap_control_log[cap][time] = num

        last_update_step = self.cap_last_update[cap]
        old_state_range = np.arange(last_update_step + 1, time)
        self.cap_control_log[cap][old_state_range] = self.cap_control_log[cap][last_update_step]

    def plot_control(self, cap):
        self.cap_control_recorder(cap, len(self.cap_control_log[cap]), 0)
        plt.figure()
        plt.plot(self.cap_control_log[cap])
        plt.title('States of capacitor %s' % cap)
        plt.show()


class RegulatorControl:
    def __init__(self, dss, total_time):
        self.state_continuity = {}
        self.tap_limit = {}
        self.transformerTap = {}
        self.tap_width = 0.00625

        for x, handle in dss.transformer_dict.items():
            if 'reg' in x:
                dss.circuit.Transformers.Name = x
                minTap = float(dss.circuit.Transformers.MinTap)
                maxTap = float(dss.circuit.Transformers.MaxTap)
                self.tap_limit[x] = [minTap, maxTap]
                self.transformerTap[x] = np.zeros(total_time)

    def observe_node_power(self, dss: DSS, bus, rc: DirectRecord):
        tap_ratio = {}

        for i in dss.bus_class_dict[bus].phase_loc:
            node = bus + '.' + str(i)
            V_pu = rc.bus_voltages[bus][i - 1, rc.current_step]
            tap_ratio['reg' + str(i)] = np.sqrt(1 / np.abs(V_pu))

        return tap_ratio

    def control_regulator(self, circuit, tap_ratio, time):
        for reg, ratio in tap_ratio.items():
            circuit.Transformers.Name = reg
            circuit.Transformers.Wdg = 2

            old_tap_ratio = circuit.Transformers.Tap
            new_tap = np.round((old_tap_ratio * ratio - 1) / self.tap_width)
            new_tap_ratio = 1 + new_tap * self.tap_width

            new_tap_ratio = max(self.tap_limit[reg][0], new_tap_ratio)
            new_tap_ratio = min(self.tap_limit[reg][1], new_tap_ratio)

            circuit.Transformers.Tap = new_tap_ratio
            new_tap = int((new_tap_ratio - 1) / self.tap_width)
            self.record(reg, new_tap, time)

    def record(self, reg, tap, time):
        self.transformerTap[reg][time] = tap