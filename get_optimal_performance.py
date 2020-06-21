import numpy as np
import pandas as pd
from src.supervised_learning.fabricating.fabricate_features import metric, name_generator
import json
import pickle
import matplotlib.pyplot as plt


class BestActionPerformance:
    def __init__(self, data: dict, metric_ratio, eval_bus, eval_phase):
        self.metric_ratio = metric_ratio if metric_ratio is not None else 2
        self.total_step = list(list(data.get(0).values())[0].values())[0].shape[1]
        self.class_num = len(data)
        self.min_tap = min(list(data.keys()))
        self.bus = eval_bus
        self.phase = eval_phase
        self.data = {tap: v[self.bus][self.phase] for (tap, v) in data.items()}

    def get_opt_v(self, shift, interval: int = 10):
        vt = np.zeros(self.total_step, dtype=complex)
        opt_tap_vec = np.zeros(self.total_step, dtype=int)

        segment = int((self.total_step - shift) / interval)
        if (self.total_step - shift) % interval:
            segment += 1
        if shift:
            segment += 1
        
        if shift:
            opt_tap = self.get_optimal_tap(0, shift)
            vt[0: shift] = self.data[opt_tap][0, 0: shift]
            opt_tap_vec[0: shift] = opt_tap
            
        for i in range(0, segment):
            start = min(i * interval + shift, self.total_step)
            end = min(start + interval, self.total_step)
            if start == end:
                continue

            opt_tap = self.get_optimal_tap(start, end)
            vt[start: end] = self.data[opt_tap][0, start: end]
            opt_tap_vec[start: end] = opt_tap
            
        return vt, opt_tap_vec

    def get_optimal_tap(self, start, end):
        vec = np.zeros(self.class_num)
        for tap, v in self.data.items():
            idx = tap - self.min_tap
            v_amp = np.abs(v[0, start: end])
            evaluation = [f(v_amp) for f in metric]
            res = evaluation[0]
            if len(metric) > 1:
                for i in range(1, len(metric)):
                    res += self.metric_ratio * evaluation[i]
            vec[idx] = res
        return np.argmin(vec) + self.min_tap
    
    
if __name__ == "__main__":
    with open('parameter_setting/get_optimal_performance.json') as r:
        par = json.load(r)
        root = par['root']
        data_path = root + par['data']
        eval_loc = par['eval_loc'].split('.')
        eval_bus = eval_loc[0]
        eval_phase = int(eval_loc[1])

        with open(data_path, 'rb') as read_dict:
            data = pickle.load(read_dict)

            extract_best = BestActionPerformance(data, par.get("metric_ratio"), eval_bus, eval_phase)
            interval = par["model_parameter"]["interval"]
            shift = par["model_parameter"]["shift"]

            for s in shift:
                time_series, taps = extract_best.get_opt_v(s, interval)
                name = name_generator(None, 10, s, 'pkl')
                with open(root + par.get('result_dir') + name, 'wb') as w:
                    pickle.dump(time_series, w)
                    pickle.dump(taps, w)

                fig, axes = plt.subplots(2, 1)
                fig.suptitle("Shift %s, interval %s" % (s, interval), y=1)
                axes[0].plot(np.abs(time_series))
                axes[0].set_ylim(0.87, 1.13)
                axes[0].set_xlabel('time')
                axes[0].set_ylabel('voltage')
                axes[1].plot(taps)
                axes[1].set_ylim(-17, +17)
                axes[1].set_xlabel('time')
                axes[1].set_ylabel('Optimal taps')

                fig.show()