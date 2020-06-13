from .circuit.CircuitComponents import DSS
from .circuit.LoadProfile import LoadVariation
from .circuit.CircuitRecorder import DirectRecord, RecordNode
from .circuit.ControlLoop import CapacitorControl, RegulatorControl
from .circuit.Storage import storage_monitor
from .circuit.PVSystem import PVSystem, Cloud_covering
from .utils.time import *
import os
import numpy as np


class Simulation:
    def __init__(self, circuit_path: str, root: str, time_span: str, step: str):
        self.time_span = time_span
        self.step_size = step
        self.total_step = get_total_step(time_span, step)
        self.d = DSS(root + circuit_path)
        self.root = root

        self.simulation_mode = None
        if 'y' not in time_span:
            self.simulation_mode = 'daily'
        else:
            self.simulation_mode = 'yearly'

        self.d.text.Command = 'set mode=%s' % self.simulation_mode
        self.d.text.Command = 'set stepsize=%s' % self.step_size
        self.d.solution.Number = 1

        self.simulated = False
        self.observation_loc = []
        self.disable_list = []
        self.enable_list = []

    def show_all_observation_loc(self):
        print('Nodes in the circuit: ')
        for i in self.d.bus_class_dict.keys():
            print(i)
        print('The first one is selected for regulator control')

    def add_observation_at_bus(self, bus):
        if type(bus) is str:
            self.observation_loc.append(bus)
        elif type(bus) is list:
            self.observation_loc += bus

    def switch_on_off(self, enable: bool = False):
        elem_list = self.enable_list if enable else self.disable_list
        output_str = 'enabled.' if enable else 'disabled.'

        for ele_dict in elem_list:
            for x, id in ele_dict.items():
                print(self.d.circuit.CktElements(id).Name + " " + output_str)
                self.d.circuit.CktElements(id).Enabled = enable

    def stop(self):
        self.d.release_com()


class BaseSimulation(Simulation):
    def __init__(self, circuit_path: str, root: str, time_span: str = '1d', step: str = '1m',
                 cap_control: str = 'none', reg_control: str = 'none', with_pv:bool = False):
        super(BaseSimulation, self).__init__(circuit_path, root, time_span, step)

        self.rc = DirectRecord(self.d, self.total_step)
        self.cap_ctrl = None
        self.reg_ctrl = None
        self.pv = None

        # Set up mode to control regulators
        if reg_control in ['inside', 'auto']:
            reg_name = self.d.find_regulator()
            self.reg_ctrl = RegulatorControl(self.d, reg_name, reg_control)      # when 'auto', record
            print('Name of regulators in the circuit: %s' % reg_name)
        if reg_control in ['inside', 'none']:
            self.disable_list.append(self.d.regcontrol_dict)

        # Set up mode to control capacitors
        if cap_control in ['inside', 'auto']:
            self.cap_ctrl = CapacitorControl(self.d, cap_control)     # for recording the control action of cap_control
        elif cap_control == 'none':
            self.disable_list.append(self.d.capcontrol_dict)
            self.disable_list.append(self.d.capacitor_dict)
            # self.d.text.Command = 'calcv' # Careful! this command changes the index of buses (or maybe elements)
            # after disabling the components, solve again, bus handles change. Need another bus handle assignment
            # Problem solved by using Bus Name to get data rather than element handles.

        # Set up PV
        if with_pv:
            self.pv = PVSystem(self.d, self.total_step)
            self.pv.set_profile(root=self.root)
        else:
            self.disable_list.append(self.d.pv_dict)

        self.switch_on_off(enable=False)

    def run(self):
        loading = LoadVariation(self.d.load_dict.keys(), root=self.root)

        for step in range(self.total_step):
            loading.load_circuit(self.d.circuit, step, actual=False)
            if self.pv is not None:
                self.pv.load_pv(self.d.circuit, step)

            self.d.solution.Solve()
            self.rc.record(self.d, step)

            if self.pv is not None:
                self.pv.record_pv(self.d.circuit, step)
            # battery_record.record(d.circuit, step)

            if self.cap_ctrl is not None:
                if self.cap_ctrl.external_control:
                    observation = self.cap_ctrl.observe_node_power(self.d, self.observation_loc, self.rc)
                    self.cap_ctrl.control_action(self.d, observation, step, mode='power')
                else:
                    # TODO: record auto cap_control actions
                    pass  # add independent observation

            if self.reg_ctrl is not None:
                if self.reg_ctrl.external_control:
                    reg_observe = self.reg_ctrl.observe_node_power(self.d, self.observation_loc[0], self.rc)
                    self.reg_ctrl.control_regulator(self.d.circuit, reg_observe, step)
                else:
                    self.reg_ctrl.record_auto_tap(self.d, self.rc)


class ExtractFeature(BaseSimulation):
    def __init__(self, circuit_path: str, root: str, window_len: int = 10, time_span: str = '1d', step: str = '1m',
                 mode: str = 'reg', cap_control: str = 'none', reg_control: str = 'none', with_pv:bool = False,
                 observation: str = '670', phase: int = 1):
        super(ExtractFeature, self).__init__(circuit_path=circuit_path,
                                             root=root,
                                             time_span=time_span,
                                             step=step,
                                             cap_control=cap_control,
                                             reg_control=reg_control,
                                             with_pv=with_pv)
        # TODO: battery scenario
        del self.rc
        assert observation in self.d.bus_class_dict.keys(), "Observation location does not exist!"
        self.observation = observation
        self.phase = phase
        self.window = window_len
        self.mode = mode
        self.disable_list = []

        if self.mode == 'reg':
            self.disable_list.append(self.d.regcontrol_dict)
            self.reg_name = self.d.find_regulator()
            self.reg_ctrl = RegulatorControl(self.d, self.reg_name, 'inside')  # when 'auto', record
            print('Name of regulators in the circuit: %s' % self.reg_name)

        elif self.mode == 'cap':
            self.enable_list.append(self.d.capacitor_dict)
            self.disable_list.append(self.d.capcontrol_dict)
            self.cap_ctrl = CapacitorControl(self.d, 'inside')

        for status in [True, False]:
            self.switch_on_off(enable=status)

    def extract(self, storage_path: str, window: int = 10):
        loading = LoadVariation(self.d.load_dict.keys(), root=self.root)

        segments = int(self.total_step / window) if not self.total_step % window else int(self.total_step / window) + 1
        action_range = None
        element = None
        if self.mode == 'reg':
            element = self.reg_ctrl.reg_name + str(self.phase)
            range_limit = self.reg_ctrl.tap_range[element]
            action_range = np.arange(range_limit[0], range_limit[1] + 1, step=1)
        elif self.mode == 'cap':
            pass

        record = {action: np.zeros((2, self.total_step), dtype=complex) for action in action_range}
        node_recorder = RecordNode(self.d, self.observation, phase=self.phase)

        for segment in range(segments):
            for tap in action_range:
                if self.mode == "reg":
                    self.reg_ctrl.set_tap(self.d, element, tap)
                elif self.mode == "cap":
                    pass

                for step in range(segment * window, min((segment+1) * window, self.total_step)):
                    loading.load_circuit(self.d.circuit, step, actual=False)        # TODO: merge loading?
                    if self.pv is not None:
                        self.pv.load_pv(self.d.circuit, step)

                    self.d.solution.Solve()
                    record[tap][:, step] = node_recorder.fetch(self.d)

                    if self.pv is not None:
                        self.pv.record_pv(self.d.circuit, step)

                    if self.cap_ctrl is not None and self.mode != 'cap':
                        if not self.cap_ctrl.external_control:
                            # TODO: record auto cap_control actions
                            pass  # add independent observation

                    if self.reg_ctrl is not None and self.mode != 'reg':
                        if self.reg_ctrl.external_control:
                            reg_observe = self.reg_ctrl.observe_node_power(self.d, self.observation_loc[0], self.rc)
                            self.reg_ctrl.control_regulator(self.d.circuit, reg_observe, step)
                        else:
                            self.reg_ctrl.record_auto_tap(self.d, self.rc)
