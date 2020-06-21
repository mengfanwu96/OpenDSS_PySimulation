from src.Simulations import *
import os
import json
import pickle


if __name__ == '__main__':
    sim = None

    # loading simulation parameters
    with open("parameter_setting/ExtractFeature_parameter.json") as par:
        pars = json.load(par)
        sim_pars = pars['simulation_parameter']
        root = pars['root']
        model_script = pars['model_script']
        storage_path = pars['storage_path']

        record_loc = pars['observe_loc']
        mode_par = pars['mode']
        mode = list(mode_par.keys())[0]
        to_reg = mode_par[mode]

        sim = ExtractFeature(circuit_path=model_script,
                             root=root,
                             time_span=sim_pars['duration'],
                             step=sim_pars['step'],
                             cap_control=sim_pars['cap_control'],
                             reg_control=sim_pars['reg_control'],
                             with_pv=sim_pars['with_pv'],
                             mode=mode,
                             to_ctrl=to_reg)

        data = sim.extract(record_loc, to_reg)

        with open(root + storage_path + "extracted%s.pkl" % (record_loc), "wb") as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
