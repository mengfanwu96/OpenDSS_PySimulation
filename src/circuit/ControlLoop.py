import numpy as np
from collections import Counter
from .CircuitComponents import DSS
from .CircuitRecorder import DirectRecord, RecordNode
from ..utils.time import *
from ..supervised_learning.fabricate_features import FabricateFeature
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib


def not_observe(time: int, window: int, shift: int):
    return bool( (time - shift) % window )


class CapacitorControl:
    def __init__(self, dss, mode: str):
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
        self.external_control = True if mode == 'inside' else False

    def capacitor_init(self, circuit, cap_list, code=None):
        # some problem: initial state can not be specified per capacitor
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
                kVA = np.abs(V * I)     # TODO: any method to directly obtain power? ActiveCktElement.Powers
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
    def __init__(self, dss, reg_name, parameter, time_span:str, step_size:str, root:str):
        self.state_continuity = {}
        self.tap_ratio_limit = {}
        self.tap_num_range = {}
        self.tap_width = {}
        self.reg_name = reg_name
        self.root = root

        self.transformerTap = {(reg_name + str(i)): np.zeros(get_steps(time_span, step_size))
                               for i in [1, 2, 3]}
        self.last_observe = {(reg_name + str(i)): -1
                               for i in [1, 2, 3]}
        self.record_tap_parameter(dss)
        self.controller = {i: {} for i in range(1, 4)}
        self.step_size = step_size

        self.active_reg_control = []
        if type(parameter) is dict:
            self.set_observation_loc_parameter(parameter, dss.bus_class_dict)
        elif type(parameter) is list:
            for i in parameter:
                self.set_observation_loc_parameter(i, dss.bus_class_dict)

    def set_observation_loc_parameter(self, p: dict, bd:dict):
        if p['mode'] == 'none':
            assert "phase" in p.keys(), "Phase for disabling regulator control not specified."
            invalid_phase = None
            if type(p['phase']) is int:
                invalid_phase = [p['phase']]
            elif type(p['phase']) is list:
                invalid_phase = p['phase']
            for i in invalid_phase:
                none_mode = {'mode': 'none'}
                if self.check_controller_to_write(i, none_mode):
                    self.controller[i] = none_mode
        else:
            bus_phase = p['base_loc'].split('.')
            bus = bus_phase[0]
            del bus_phase[0]
            phase = [int(i) for i in bus_phase] if len(bus_phase) else bd[bus].phase_loc

            for i in phase:
                if self.check_controller_to_write(i, p):
                    self.controller[i] = {'bus': bus,
                                          'mode': p['mode'],
                                          'interval': get_steps(p['interval'], self.step_size),
                                          'shift': get_steps(p['shift'], self.step_size, pos=False)}
                    self.controller[i]['shift'] %= self.controller[i]['interval']
                    if self.controller[i]['mode'] == 'sl':
                        assert 'model' in p.keys(), "Supervised-learning model not found!"
                        self.controller[i]['model'] = self.get_trained_supervised_model(self.root + p['model'])
                        self.controller[i]['feature_fabricator'] = FabricateFeature(self.controller[i]['interval'])

    def get_active_auto_control(self):
        res = []
        for phase, controller in self.controller.items():
            if controller['mode'] == 'auto':
                res.append(phase)
        return res

    def remove_none_controller(self):
        to_delete = []
        for phase, controller in self.controller.items():
            if controller and controller['mode'] == 'none':
                to_delete.append(phase)
            elif not controller:
                to_delete.append(phase)

        for i in to_delete:
            del self.controller[i]

    def add_node_recorder(self, dss: DSS, rc: DirectRecord):
        for phase, controller in self.controller.items():
            if controller['mode'] == 'inside' or controller['mode'] == 'sl':
                rc.add_node_recorder(dss, controller['bus'])

    def check_controller_to_write(self, phase: int, new_setting: dict):
        if self.controller[phase]:   # if the controller is already set
            print("Duplicated controller setting for phase %s" % phase)
            print("Original setting: %s" % self.controller[phase])
            print("New setting: %s" % new_setting)
            override = bool(input("Override? 1: yes, 0: no"))
            return override
        else:
            return True

    def record_tap_parameter(self, dss: DSS):
        for x, handle in dss.transformer_dict.items():
            if self.reg_name in x:
                dss.circuit.Transformers.Name = x
                minTap = float(dss.circuit.Transformers.MinTap)
                maxTap = float(dss.circuit.Transformers.MaxTap)
                tap_num = int(dss.circuit.Transformers.NumTaps)
                self.register_tap_parameter(x, minTap, maxTap, tap_num)

    def register_tap_parameter(self, reg, min_tap, max_tap, tap_num):
            self.tap_ratio_limit[reg] = [min_tap, max_tap]
            width = (max_tap - min_tap) / tap_num
            self.tap_width[reg] = width
            self.tap_num_range[reg] = [np.round((Tap - 1) / width).astype(int) for Tap in self.tap_ratio_limit[reg]]

    def set_tap_parameter(self, dss: DSS, phase, num_taps: int, max_tap: float = 1.1, min_tap: float = 0.9):
        phase_list = list(phase)
        for i in phase_list:
            x = self.reg_name + str(i)
            dss.circuit.Transformers.Name = x
            dss.circuit.Transformers.NumTaps = num_taps
            dss.circuit.Transformers.MaxTap = max_tap
            dss.circuit.Transformers.MinTap = min_tap
            self.register_tap_parameter(x, min_tap, max_tap, num_taps)

    def set_tap(self, circuit, reg: str, tap: int):
        ratio = self.tap2ratio(tap, reg=reg)
        circuit.Transformers.Name = reg
        circuit.Transformers.Tap = ratio

    def ratio2tap(self, ratio: float, **kwargs) -> int:
        assert 'phase' in kwargs.keys() or 'reg' in kwargs.keys(), "Regulator index not specified."
        if 'phase' in kwargs.keys() and 'reg' not in kwargs.keys():
            reg = self.reg_name + str(kwargs['phase'])
        else:
            reg = kwargs['reg']

        assert self.tap_ratio_limit[reg][0] <= ratio <= self.tap_ratio_limit[reg][1], "Tap scaling ratio not in range."

        k = np.round((ratio - 1) / self.tap_width[reg])
        return k

    def tap2ratio(self, tap: int, **kwargs) -> float:
        assert 'phase' in kwargs.keys() or 'reg' in kwargs.keys(), "Regulator index not specified."
        if 'phase' in kwargs.keys() and 'reg' not in kwargs.keys():
            reg = self.reg_name + str(kwargs['phase'])
        else:
            reg = kwargs['reg']

        assert self.tap_num_range[reg][0] <= tap <= self.tap_num_range[reg][1], "Tap number not in range."

        ratio = tap * self.tap_width[reg] + 1
        return ratio

    def observe_node(self, rc: DirectRecord):
        observation = {}
        for phase, controller in self.controller.items():
            a = None
            if controller['mode'] in ['auto', 'none'] or \
                    not_observe(rc.current_step, controller['interval'], controller['shift']):
                pass
            elif controller['mode'] == 'inside':
                a = self.observe_node_power(controller['bus'], phase, rc)
            elif controller['mode'] == 'sl':
                a = self.get_sl_feature(controller['bus'], phase, rc)
            observation[phase] = a
        return observation

    def get_sl_feature(self, bus:str, phase:int, rc: DirectRecord):
        v_vector = rc.bus_voltages[bus][phase - 1, :]
        c_vector = rc.line_currents[rc.node_recorder[bus].line.name][phase - 1, :]
        tap = self.transformerTap[self.reg_name + str(phase)][rc.current_step - 1]
        fv = self.controller[phase]['feature_fabricator'].fabricate_feature_vector(v_vector, c_vector, rc.current_step, tap)

        return fv

    def observe_node_power(self, bus: str, phase: int, rc: DirectRecord):
        tap_ratio = {}
        V_pu = rc.bus_voltages[bus][phase - 1, rc.current_step]
        tap_ratio[self.reg_name + str(phase)] = np.sqrt(1 / np.abs(V_pu))
        # TODO: try direct reciprocate with out square root?

        return tap_ratio

    def control_regulator(self, circuit, observation, time):
        for phase, controller in self.controller.items():
            action = None
            record = None

            if controller['mode'] == 'auto':
                record = self.get_auto_tap(circuit, phase)
            elif observation[phase] is not None:
                if controller['mode'] == 'inside':
                    action = self.scaling_based_control(circuit, observation[phase])
                elif controller['mode'] == 'sl':
                    action = self.supervised_learning_control(phase, observation[phase])

            if action is not None:
                self.set_tap(circuit, self.reg_name + str(phase), action)
                self.record(self.reg_name + str(phase), action, time)
            if record is not None:
                self.record(self.reg_name + str(phase), record, time)

    def scaling_based_control(self, circuit, tap_ratio: dict) -> int:
        for reg, ratio in tap_ratio.items():
            circuit.Transformers.Name = reg
            circuit.Transformers.Wdg = 2

            old_tap_ratio = circuit.Transformers.Tap
            new_tap_ratio = old_tap_ratio * ratio
            new_tap_ratio = max(self.tap_ratio_limit[reg][0], new_tap_ratio)
            new_tap_ratio = min(self.tap_ratio_limit[reg][1], new_tap_ratio)

            new_tap = self.ratio2tap(ratio=new_tap_ratio, reg=reg)
            return new_tap

    def supervised_learning_control(self, phase: int, observation:np.ndarray) -> int:
        return int(self.controller[phase]['model'].predict(observation))

    def record(self, reg: str, tap: int, time: int):
        self.transformerTap[reg][time] = tap

        if time > self.last_observe[reg] + 1:
            self.transformerTap[reg][self.last_observe[reg] + 1: time] = \
                self.transformerTap[reg][self.last_observe[reg]]
        self.last_observe[reg] = time

    def get_auto_tap(self, circuit, phase: int):
        reg = self.reg_name + str(phase)
        circuit.Transformers.Name = reg
        circuit.Transformers.Wdg = 2
        tap_ratio = circuit.Transformers.Tap

        return self.ratio2tap(tap_ratio, reg=reg)

    def plot(self, reg=None):  # TODO: selective plotting
        fig, axes = plt.subplots(len(self.transformerTap), 1)
        if len(self.transformerTap) == 1:
            axes = [axes]

        for idx, (x, data) in enumerate(self.transformerTap.items()):
            axes[idx].plot(data)
            axes[idx].set_xlabel('time')
            axes[idx].set_ylabel('controller Tap')

        fig.show()

    @staticmethod
    def get_trained_supervised_model(path:str):
        return joblib.load(path)
