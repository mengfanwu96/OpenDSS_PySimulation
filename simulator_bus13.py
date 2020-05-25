from LoadProfile import LoadVariation
from CircuitRecorder import DirectRecord
from CircuitComponents import DSS
from ControlLoop import CapacitorControl, RegulatorControl
from Storage import storage_monitor
from PVSystem import PVSystem
import os
import pickle
import re


if __name__ == '__main__':
    time_span = 1 # in days
    step_size = "1m"  # format: (n, unit)
    total_step = 0
    unit_dict = {'d':1, 'h':24, 'm':60*24, 's':3600*24}

    num_unit = re.split('([dhms])', step_size)
    assert '0' <= num_unit[0] <= '9', "number is mandatory"
    assert num_unit[1] in unit_dict.keys(), "unit not found"
    total_step = int(time_span * unit_dict[num_unit[1]] / int(num_unit[0]))
    assert total_step >= 1, "time span shorter than step size"

    d = DSS(os.getcwd() + "\IEEE13Nodeckt.dss")
    d.text.Command = "set mode=daily"
    d.text.Command = "set stepsize=%s" % step_size
    d.solution.Number = 1
    overwrite = True
    buses_to_regulate = ['675']

    if overwrite:
        loading = LoadVariation(d.load_dict.keys())
        rc = DirectRecord(d, total_step)
        cap_ctrl = CapacitorControl(d)
        reg_ctrl = RegulatorControl(d)
        # battery_record = storage_monitor(d, total_step)
        pv = PVSystem(d, total_step)
        pv.set_profile()

        for step in range(total_step):
            loading.load_circuit(d.circuit, step, actual=False)
            pv.load_pv(d.circuit, step)
            d.solution.Solve()
            rc.record(d, step)
            pv.record_pv(d.circuit, step)
            # battery_record.record(d.circuit, step)
            observation = cap_ctrl.observe_node_power(d, buses_to_regulate, rc)
            reg_observe = reg_ctrl.observe_node_power(d, '670', rc)
            cap_ctrl.control_action(d, observation, step, mode='power')
            reg_ctrl.control_regulator(d.circuit, reg_observe, step)

        with open("circuit_record_ctrl_2.0.pkl", "wb") as output:
            pickle.dump((rc, loading, cap_ctrl, reg_ctrl, pv), output, pickle.HIGHEST_PROTOCOL)

    else:
        with open("circuit_record_ctrl_2.0.pkl", "rb") as read:
            rc, loading, cap_ctrl, reg_ctrl, pv = pickle.load(read)

    pv.plot_pv()

    for i in ['675', '684']:
        rc.plot_busV(i)
        # loading.plot_load_onBus(i, d.circuit)
    # for i in cap_ctrl.cap_control_log.keys():
    #     cap_ctrl.plot_control(i)
    # reg_ctrl.plot()

