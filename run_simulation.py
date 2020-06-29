from src.Simulations import *
import os
import json
import pickle

if __name__ == '__main__':
    sim = None

    # loading simulation parameters
    with open("parameter_setting/serial_parameter.json") as par:
        pars = json.load(par)
        sim_pars = pars['simulation_parameter']
        to_plot = pars['plot']
        root = pars['root']
        model_script = pars['model_script']
        pickle_path = root + pars['pickle_path']

        # read from stored data
        for a in ['0', '0_3', '0_2_5', '0_2_4_6', '0_1_2_3_4_5_6_7_8']:
            reg_sim = sim_pars['reg_control'][0]
            reg_sim['model'] = "data/supervised_learning/model/w10_int10_shift_" + a + '.pkl'
            if not pars['override']:
                with open(pickle_path + "circuit_record.pkl", "rb") as read:
                    sim = pickle.load(read)
            # or new simulation
            else:
                sim = BaseSimulation(circuit_path=model_script,
                                     root=root,
                                     time_span=sim_pars['duration'],
                                     step=sim_pars['step'],
                                     cap_control=sim_pars['cap_control'],
                                     reg_control=reg_sim,
                                     with_pv=sim_pars['with_pv'])
                sim.run()

                # stop the COM interface so that we can store the data
                sim.stop()
                with open(pickle_path + "circuit_record.pkl", "wb") as output:
                    pickle.dump(sim, output, pickle.HIGHEST_PROTOCOL)

            # Plotting
                bus_to_plot = to_plot['bus']
                if type(bus_to_plot) is not list:
                    bus_to_plot = [bus_to_plot]
                for i in bus_to_plot:
                    sim.rc.plot_busV(i)

            if to_plot['reg_tap']:
                sim.reg_ctrl.plot()



