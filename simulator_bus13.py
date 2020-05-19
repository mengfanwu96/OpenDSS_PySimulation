from LoadProfile import LoadVariation
from CircuitRecorder import DirectRecord
from CircuitComponents import DSS
from ControlLoop import CapacitorControl, RegulatorControl
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt


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
    overwrite = True
    buses_to_regulate = ['675', '684']

    if overwrite:
        loading = LoadVariation(d.load_dict.keys())
        rc = DirectRecord(d, total_step)
        cap_ctrl = CapacitorControl(d)
        # reg_ctrl = RegulatorControl(d, 1440)

        for step in range(1440):
            loading.load_circuit(d.circuit, step, actual=False)
            d.solution.Solve()
            rc.record(d, step)

            observation = cap_ctrl.observe_node_power(d, buses_to_regulate, rc)
            # reg_observe = reg_ctrl.observe_node_power(d, '670', rc)
            cap_ctrl.control_action(d, observation, step, mode='power')
            # reg_ctrl.control_regulator(d.circuit, reg_observe, step)

        # with open("circuit_record_ctrl_2.0.pkl", "wb") as output:
        #     pickle.dump((rc, loading, cap_ctrl, reg_ctrl), output, pickle.HIGHEST_PROTOCOL)

    else:
        with open("circuit_record_ctrl_2.0.pkl", "rb") as read:
            rc, loading, cap_ctrl, reg_ctrl = pickle.load(read)

    for i in ['675', '692', '652', '632']:
        rc.plot_busV(i)
        # loading.plot_load_onBus(i, d.circuit)
    # for i in cap_ctrl.cap_control_log.keys():
    #     cap_ctrl.plot_control(i)

    reg_list = ['reg1', 'reg2', 'reg3']
    for i in range(3):
        data1 = np.abs(rc.bus_voltages['675'][i, :])
        data2 = reg_ctrl.transformerTap[reg_list[i]]
        fig, ax1 = plt.subplots(figsize=(21, 8))

        color = 'tab:red'
        ax1.set_xlabel('time (m)')
        ax1.set_ylabel('Vol', color=color)
        ax1.plot(data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('regController Tap', color=color)  # we already handled the x-label with ax1
        ax2.plot(data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

