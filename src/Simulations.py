from .circuit.CircuitComponents import DSS
from .circuit.LoadProfile import LoadVariation
from .circuit.CircuitRecorder import DirectRecord
from .circuit.ControlLoop import CapacitorControl, RegulatorControl
from .circuit.Storage import storage_monitor
from .circuit.PVSystem import PVSystem, Cloud_covering
from .utils.time import *
import os

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


class BaseSimulation(Simulation):
    def __init__(self, circuit_path: str, root: str, time_span: str='1d', step: str='1m',
                 cap_control: str = 'none', reg_control: str = 'none', with_pv:bool = False):
        super(BaseSimulation, self).__init__(circuit_path, root, time_span, step)

        self.rc = DirectRecord(self.d, self.total_step)
        self.cap_ctrl= None
        self.reg_ctrl = None
        self.pv = None

        if reg_control in ['inside', 'auto']:
            self.reg_name = self.d.find_regulator()
            self.reg_ctrl = RegulatorControl(self.d, self.reg_name, reg_control)      # when 'auto', record
            enable_reg_control = True if reg_control == 'auto' else False
            for x, id in self.d.regcontrol_dict.items():
                self.d.circuit.CktElements(id).Enabled = enable_reg_control

            if reg_control == 'inside':
                self.reg_name = self.d.find_regulator()
                print('Name of regulators in the circuit: %s' % self.reg_name)

        elif reg_control == 'none':
            for x, id in self.d.regcontrol_dict.items():
                self.d.circuit.CktElements(id).Enabled = False

        if cap_control in ['inside', 'auto']:
            self.cap_ctrl = CapacitorControl(self.d, cap_control)     # for recording the control action of cap_control
        elif cap_control == 'none':  # TODO: practically disabling capacitors, need to separate cap_control and capcitors
            for x, id in self.d.capacitor_dict.items():
                print(self.d.circuit.CktElements(id).Name)
                self.d.circuit.CktElements(id).Enabled = False
            # self.d.text.Command = 'calcv' # Careful! this command changes the index of buses handles (or maybe elements)
            # after disabling the components, solve again, bus handles change. Need another bus handle assignment
            # Problem solved by using Bus Name to get data rather than element handles.
        if with_pv:
            self.pv = PVSystem(self.d, self.total_step)
            self.pv.set_profile(root=self.root)

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
                if self.cap_ctrl.control:
                    observation = self.cap_ctrl.observe_node_power(self.d, self.observation_loc, self.rc)
                    self.cap_ctrl.control_action(self.d, observation, step, mode='power')
                else:
                    pass  # add independent observation

            if self.reg_ctrl is not None:
                if self.reg_ctrl.control:
                    reg_observe = self.reg_ctrl.observe_node_power(self.d, self.observation_loc[0], self.rc)
                    self.reg_ctrl.control_regulator(self.d.circuit, reg_observe, step)
                else:
                    self.reg_ctrl.record_auto_tap(self.d, self.rc)

    def stop(self):
        self.d.release_com()
