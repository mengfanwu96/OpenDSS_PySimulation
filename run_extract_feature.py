from src.Simulations import *
import os
import json
import pickle


if __name__ == '__main__':
    sim = None

    # loading simulation parameters
    with open("ExtractFeature_parameter.json") as par:
        pars = json.load(par)
        sim_pars = pars['simulation_parameter']
        root = pars['root']
        model_script = pars['model_script']
        storage_path = pars['storage_path']
        observation_loc = pars['observation_ctrl']
        record_loc = pars['record_loc']
        phase = pars['phase']
        mode = pars['mode']

        sim = ExtractFeature(circuit_path=model_script,
                             root=root,
                             window_len=sim_pars['window'],
                             time_span=sim_pars['duration'],
                             step=sim_pars['step'],
                             cap_control=sim_pars['cap_control'],
                             reg_control=sim_pars['reg_control'],
                             with_pv=sim_pars['with_pv'],
                             record_loc=record_loc,
                             phase=phase,
                             mode=mode)

        sim.add_observation_at_bus(observation_loc)
        data = sim.extract()

        with open(root + storage_path + "extracted.pkl", "wb") as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
