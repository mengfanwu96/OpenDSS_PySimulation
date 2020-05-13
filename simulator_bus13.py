from LoadProfile import LoadVariation
from CircuitRecorder import DirectRecord
from CircuitComponents import DSS
from ControlLoop import *
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
    overwrite = True

    if overwrite:
        loading = LoadVariation(d.load_dict.keys())
        # rc = CircuitRecorder(d.circuit, total_step)   # use variables for system parameters
        rc = DirectRecord(d, total_step)
        cap_ctrl = capacitor_ctrl(d)
        # max_iter = 20

        capacitor_init(d.circuit, list(d.capacitor_dict.keys()))

        for step in range(1440):
            loading.load_circuit(d.circuit, step, actual=False)
            d.solution.Solve()
            rc.record(d, step)

            # actions = observe_circuit_current(d, line=['670671', '692675'])
            # actions = observe_circuit_voltage(d, bus=['671', '675'])
            # for obj, code in actions.items():
            #     capacitor_controller(d, obj, code, cap_ctrl)


        with open("circuit_record_ctrl1.0.pkl", "wb") as output:
            pickle.dump((rc, loading), output, pickle.HIGHEST_PROTOCOL)

    else:
        with open("circuit_record_ctrl1.0.pkl", "rb") as read:
            rc, loading = pickle.load(read)

    # rc.plot(bus_indexes=[7, 8, 11, 12, 16])
    # rc.plot(branch_indexes=[1])
    for i in ['675', '692', '671', '670', '652']:
        rc.plot_busV(i)
        # loading.plot_load_onBus(i, d.circuit)
    # for i in ['692675', '671684', '670671']:
    #     rc.plot_lineC(i)

