import numpy as np
import pandas as pd
import pickle
import json


# Parameters
class FabricateFeature:
    def __init__(self, window: int = 10, metric_ratio=2.0):
        self.window = window
        self.metric_ratio = metric_ratio
        self.v_feature_name = ['v%s' % i for i in range(window)]
        self.c_feature_name = ['c%s' % i for i in range(window)]
        self.angle_feature_name = ['theta%s' % i for i in range(window)]
        self.feature_names = self.v_feature_name + self.c_feature_name + self.angle_feature_name + ['tap']
        self.metric = [lambda y: np.abs(np.mean(y) - 1), np.std]
        self.metric_ratio = metric_ratio
        # TODO: maybe other complicated metric calculation

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["metric"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.metric = [lambda y: np.abs(np.mean(y) - 1), np.std]

    def fabricate_feature_vector(self, v, c, final_step:int, current_tap: int):
        if final_step - self.window < 0:
            return None
        v_vector = np.abs(v[final_step - self.window: final_step])
        c_vector = np.abs(c[final_step - self.window: final_step])
        angle_vector = np.angle(v[final_step - self.window: final_step])

        return np.concatenate((v_vector, c_vector, angle_vector, np.array([current_tap]))).reshape(1, -1)

    def get_metric_stat(self, v):
        return [f(v) for f in self.metric]

    def evaluate_metric(self, metric_stat:np.array):
        if type(self.metric_ratio) in [float, int] and len(metric_stat) == 2:
            return metric_stat[0] + self.metric_ratio * metric_stat[1]
        elif type(self.metric_ratio) is list and len(self.metric_ratio) == len(metric_stat) - 1:
            a = metric_stat[0]
            for idx, k in enumerate(self.metric_ratio):
                a += metric_stat[idx + 1] * k
            return a
        else:
            assert False, "Error in computing metrics, ratios: %s and statistics: %s" % (self.metric_ratio, metric_stat)

if __name__ == '__main__':
    with open("supervised_learning.json") as par:
        pars = json.load(par)
        data_path = pars['root'] + pars['data_dir']
        window = pars['window']
        shift = pars['shift'] % window
        metric_ratio = pars['metric_ratio']
        raw_data_path = data_path + pars['raw']

        fv = FabricateFeature(window=window, metric_ratio=metric_ratio)

        with open(raw_data_path, "rb") as read:
            data = pickle.load(read)

            total_step = list(data.values())[0].shape[1]
            segment = int(total_step / fv.window) if not total_step % fv.window else \
                int(total_step / fv.window) + 1  # TODO: differentiate window side and interval length.

            class_num = len(data)
            features = []
            stat_dict = {key: [] for key in data.keys()}

            for tap, time_series in data.items():
                for i in range(1, segment):
                    # fabricate features, with loc info, later replace loc with label
                    start = min(i * fv.window + shift, total_step)
                    end = min(start + fv.window, total_step)
                    if start == end:
                        continue

                    feature_v = fv.fabricate_feature_vector(time_series[0, :], time_series[1, :], start, tap)
                    features.append(np.concatenate((feature_v, np.array([i - 1 + class_num]).reshape(1, 1)), axis=1))

                    metric_v = np.abs(time_series[0, start: end])
                    stat_dict[tap].append(fv.get_metric_stat(metric_v))

            num_points = len(features)
            features = np.array(features).reshape(num_points, -1)
            feature_pd = pd.DataFrame(data=features, columns=fv.feature_names + ['label'])

            for i in range(0, segment-1):
                min_value = 1e6
                for x, stat_series in stat_dict.items():
                    score = fv.evaluate_metric(stat_series[i])
                    if score < min_value:
                        min_value = score
                        min_tap = x
                feature_pd["label"].replace({i + class_num: min_tap}, inplace=True)
                # possible overlap of loc_index and tap_number
                # another way of indexing

            feature_pd.to_csv(data_path + "fabricated_w%s_d%s.csv" % (fv.window, shift), index=False)

