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
        self.total_step = get_steps(time_span, step)
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

        self.rc = DirectRecord(self.d, self.total_step)
        self.simulated = False
        self.observation_loc = []
        self.disable_list = []  # list of dictionaries:
        self.enable_list = []

    def show_all_observation_loc(self):
        print('Nodes in the circuit: ')
        for i in self.d.bus_class_dict.keys():
            print(i)
        print('The first one is selected for regulator control')

    def add_observation_at_bus(self, bus):
        buses = None
        if type(bus) is str:
            buses = [bus]
        elif type(bus) is list:
            buses = bus

        for i in buses:
            assert i in self.d.bus_class_dict, "Bus %s not found!" % i
            self.observation_loc.append(i)
            self.rc.add_node_recorder(self.d, i)

    def switch_on_off(self, enable: bool = False):
        elem_list = self.enable_list if enable else self.disable_list
        output_str = 'enabled.' if enable else 'disabled.'

        for ele_dict in elem_list:
            for x, id in ele_dict.items():
                print("Name of elements in dictionary: %s" % x)
                print(self.d.circuit.CktElements(id).Name + " " + output_str)
                self.d.circuit.CktElements(id).Enabled = enable

    def get_object_dict(self, obj_name: str, phase: list): # TODO: move into circuit (DSS)
        res = {}
        all_components = self.d.name_module_mapping[obj_name]
        for name, handle in all_components.items():
            bus_connected = self.d.circuit.CktElements(handle).BusNames[0]
            phase_connected = int(bus_connected.split('.')[1])
            if phase_connected in phase:
                res.update({name: handle})
        return res

    def stop(self):
        self.d.release_com()


class BaseSimulation(Simulation):
    def __init__(self, circuit_path: str, root: str, time_span: str = '1d', step: str = '1m',
                 cap_control: str = 'none', reg_control='none', with_pv:bool = False):
        super(BaseSimulation, self).__init__(circuit_path, root, time_span, step)

        self.cap_ctrl = None
        self.reg_ctrl = None
        self.pv = None

        reg_name = self.d.find_regulator()
        print('Name of regulators in the circuit: %s' % reg_name)
        self.reg_ctrl = RegulatorControl(self.d, reg_name, reg_control, time_span, step, self.root)
        self.reg_ctrl.remove_none_controller()
        self.reg_ctrl.add_node_recorder(self.d, self.rc)
        enable_reg, disable_reg = self.reg_ctrl.get_active_auto_control()
        self.enable_list.append(self.get_object_dict("RegControl", enable_reg))
        self.disable_list.append(self.get_object_dict("RegControl", disable_reg))

        # Set up mode to control capacitors
        if cap_control in ['inside', 'auto']:
            self.cap_ctrl = CapacitorControl(self.d, cap_control)     # for recording the control action of cap_control
        elif cap_control == 'none':
            self.disable_list.append(self.d.capcontrol_dict)
            self.disable_list.append(self.d.capacitor_dict)

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
                reg_observe = self.reg_ctrl.observe_node(self.rc)
                self.reg_ctrl.control_regulator(self.d.circuit, reg_observe, step)


class ExtractFeature(Simulation):
    def __init__(self, circuit_path: str, root: str, to_ctrl, time_span: str = '1d', step: str = '1m',
                 mode: str = 'reg', cap_control: str = 'none', reg_control='none', with_pv:bool = False):
        # TODO: battery scenario
        super(ExtractFeature, self).__init__(circuit_path, root, time_span, step)

        self.cap_ctrl = None
        self.reg_ctrl = None
        self.pv = None
        self.mode = mode

        reg_name = self.d.find_regulator()
        # adopting BasesSimulation here
        if mode != 'reg':
            self.reg_ctrl = RegulatorControl(self.d, reg_name, reg_control, time_span, step, self.root)
        else:
            self.reg_ctrl = RegulatorControl(self.d, reg_name, {'mode': "direct", 'phase': to_ctrl},
                                             time_span, step, root)
        self.reg_ctrl.remove_none_controller()
        self.reg_ctrl.add_node_recorder(self.d, self.rc)
        enable_reg, disable_reg = self.reg_ctrl.get_active_auto_control()
        self.enable_list.append(self.get_object_dict("RegControl", enable_reg))
        self.disable_list.append(self.get_object_dict("RegControl", disable_reg))

        if mode != 'cap':
            if cap_control in ['inside', 'auto']:
                self.cap_ctrl = CapacitorControl(self.d, cap_control)  # for recording the control action of cap_control
            elif cap_control == 'none':
                self.disable_list.append(self.d.capcontrol_dict)
                self.disable_list.append(self.d.capacitor_dict)

        if with_pv:
            self.pv = PVSystem(self.d, self.total_step)
            self.pv.set_profile(root=self.root)
        else:
            self.disable_list.append(self.d.pv_dict)

        for status in [True, False]:
            self.switch_on_off(enable=status)

    def extract(self, record_loc, to_ctrl):
        loading = LoadVariation(self.d.load_dict.keys(), root=self.root)

        action_range = None
        element = None
        reg_phase = None

        obs_loc_phase = {}
        for k in record_loc:
            self.get_observe_loc_phase(self, self.d, k, obs_loc_phase)

        if self.mode == 'reg':
            reg_phase = to_ctrl
            element = self.reg_ctrl.reg_name + str(reg_phase)
            range_limit = self.reg_ctrl.tap_num_range[element]
            action_range = np.arange(range_limit[0], range_limit[1] + 1, step=1)
        elif self.mode == 'cap':
            pass

        record = {action: {
                    loc: {
                        phase: np.zeros((2, self.total_step), dtype=complex) for phase in x
                    } for (loc, x) in obs_loc_phase.items()
                 } for action in action_range}
        node_recorder = {bus: RecordNode(self.d, bus) for bus in obs_loc_phase.keys()}

        for tap in action_range:
            if self.mode == "reg":
                self.reg_ctrl.set_tap(self.d.circuit, element, tap)
            elif self.mode == "cap":
                pass

            for step in range(self.total_step):
                loading.load_circuit(self.d.circuit, step, actual=False)
                if self.pv is not None:
                    self.pv.load_pv(self.d.circuit, step)

                self.d.solution.Solve()
                for bus, phase_list in obs_loc_phase.items():
                    r = node_recorder[bus].fetch(self.d, phase_list)
                    for idx, i in enumerate(phase_list):
                        record[tap][bus][i][:, step] = r[:, idx]

                if self.pv is not None:
                    self.pv.record_pv(self.d.circuit, step)

                if self.cap_ctrl is not None and self.mode != 'cap':
                    if not self.cap_ctrl.external_control:
                        # TODO: record auto cap_control actions
                        pass  # add independent observation

                if self.reg_ctrl is not None and self.mode != 'reg':
                    reg_observe = self.reg_ctrl.observe_node(self.rc)
                    self.reg_ctrl.control_regulator(self.d.circuit, reg_observe, step)
        return record

    @staticmethod
    def get_observe_loc_phase(self, dss: DSS, obs: str, obs_dict):
        loc_phase = obs.split('.')
        loc = loc_phase[0]
        if loc in dss.bus_class_dict:
            if len(loc_phase) > 1:
                for i in range(1, len(loc_phase)):
                    phase = int(loc_phase[i])
                    assert phase in dss.bus_class_dict[loc].phase_loc, \
                        "Bus %s loc does not have phase %s" % (loc, phase)
                    self.write_bus_obs(obs_dict, loc, int(loc_phase[i]))
            elif len(loc_phase) == 1:
                obs_dict[loc] = dss.bus_class_dict[loc].phase_loc

    @staticmethod
    def write_bus_obs(obs_dict, bus: str, phase: int):
        if bus in obs_dict:
            obs_dict[bus].append(phase)
        else:
            obs_dict[bus] = [phase]


    # TODO: check performance of one window shift in other situation? (ml control module in simulation)

