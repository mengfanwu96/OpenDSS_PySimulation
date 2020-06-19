import numpy as np
import pandas as pd
import pickle
import json


def name_generator(window, interval, shift, suffix: str, prefix: str = None):
    name = 'w%s_int%s_shift' % (window, interval)
    if prefix is not None:
        name = prefix + '_' + name

    if type(shift) is int:
        name += "_" + str(shift)
    elif type(shift) is list:
        for x in shift:
            name += '_' + str(x)
    return name + '.' + suffix


metric = [lambda y: np.abs(np.mean(y) - 1), np.std]
metric_ratio = 2

# Parameters
class FabricateFeature:
    def __init__(self, window: int = 10, metric_ratio=2.0):
        self.window = window
        self.metric_ratio = metric_ratio
        self.v_feature_name = ['v%s' % i for i in range(window)]
        self.c_feature_name = ['c%s' % i for i in range(window)]
        self.v_angle_feature_name = ['theta%s' % i for i in range(window)]
        self.c_angle_feature_name = ['phi%s' % i for i in range(window)]

        self.feature_names = self.v_feature_name + self.c_feature_name + \
                             self.v_angle_feature_name + self.c_angle_feature_name + ['tap']
        self.metric = metric
        self.metric_ratio = metric_ratio


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
        theta_vector = np.angle(v[final_step - self.window: final_step])
        phi_vector = np.angle(c[final_step - self.window: final_step])

        return np.concatenate((v_vector, c_vector, theta_vector, phi_vector, np.array([current_tap]))).reshape(1, -1)

    def get_metric_stat(self, v: np.ndarray, start: int, end: int):
        metric_v = v[start: end]
        if v.dtype is complex:
            metric_v = np.abs(metric_v)

        return [f(metric_v) for f in self.metric]

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


class Fabricator:
    def __init__(self, data: dict, metric_ratio):
        self.data = data
        self.metric_ratio = metric_ratio
        self.total_step = list(data.values())[0].shape[1]
        self.class_num = len(data)
        self.feature_extractor = None
        self.output = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["data"]
        return state

    def fabricate(self, window: int, shift: int, storage_path: str, **kwargs):
        interval = window if 'interval' not in kwargs.keys() else kwargs['interval']
        self.feature_extractor = FabricateFeature(window=window, metric_ratio=metric_ratio)

        segment = int((self.total_step - shift) / interval)
        if (self.total_step - shift) % interval:
            segment += 1
        if shift:
            segment += 1

        features = []
        stat_dict = {key: [] for key in data.keys()}

        valid_segment = None
        for tap, time_series in data.items():
            valid_segment = 0
            for i in range(1, segment):
                # fabricate features, with loc info, later replace loc with label
                start = min(i * interval + shift, self.total_step)
                end = min(start + interval, self.total_step)
                if start == end:
                    continue

                # feature vector is of window size
                feature_v = self.feature_extractor.fabricate_feature_vector(time_series[0, :], time_series[1, :], start, tap)

                if feature_v is None:
                    continue
                features.append(np.concatenate((feature_v,\
                                                np.array(valid_segment + self.class_num).reshape(1, 1)), axis=1))
                valid_segment += 1

                # evaluation vector is of interval size
                stat_dict[tap].append(self.feature_extractor.get_metric_stat(time_series[0, :], start, end))

        num_points = len(features)
        features = np.array(features).reshape(num_points, -1)
        feature_pd = pd.DataFrame(data=features, columns=self.feature_extractor.feature_names + ['label'])

        for i in range(valid_segment):
            min_value = 1e6
            min_tap = 0
            for x, stat_series in stat_dict.items():
                score = self.feature_extractor.evaluate_metric(stat_series[i])
                if score < min_value:
                    min_value = score
                    min_tap = x
            feature_pd["label"].replace({i + self.class_num: min_tap}, inplace=True)
        self.output = feature_pd


if __name__ == '__main__':
    with open("fabricate_features.json") as par:
        pars = json.load(par)
        data_path = pars['root'] + pars['data_dir']
        window = pars['window']
        shift = pars['shift']
        interval = pars['interval']
        metric_ratio = pars['metric_ratio']
        raw_data_path = data_path + pars['raw']

        with open(raw_data_path, "rb") as read:
            data = pickle.load(read)
            fabricator = Fabricator(data, metric_ratio)

            for s in shift:
                fabricator.fabricate(window, s, data_path, interval=interval)
                to_save = [fabricator.output, {"window": window, "shift": s, "interval": interval}]
                with open(data_path + name_generator(window, interval, s, 'pkl', prefix='fabricated'), 'wb') as w:
                    for i in to_save:
                        pickle.dump(i, w)
