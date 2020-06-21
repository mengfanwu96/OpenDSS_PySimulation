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


def get_bus_phase(bus: str):
    bus_loc = bus.split('.')
    rd = {}
    if len(bus_loc) == 1:
        rd[bus_loc[0]] = 'all'
    else:
        rd[bus_loc[0]] = []
        for i in range(1, len(bus_loc)):
            rd[bus_loc[0]].append(int(bus_loc[i]))
    return rd


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
                             self.v_angle_feature_name + self.c_angle_feature_name
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

    def fabricate_feature_vector(self, v, c, final_step:int, **kwargs):
        if final_step - self.window < 0:
            return None
        v_vector = np.abs(v[final_step - self.window: final_step])
        c_vector = np.abs(c[final_step - self.window: final_step])
        theta_vector = np.angle(v[final_step - self.window: final_step])
        phi_vector = np.angle(c[final_step - self.window: final_step])

        tap = kwargs.get('current_tap')
        if tap is not None:
            return np.concatenate((v_vector, c_vector, theta_vector, phi_vector, np.array([tap]))).reshape(1, -1)
        else:
            return np.concatenate((v_vector, c_vector, theta_vector, phi_vector)).reshape(1, -1)

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
        self.total_step = list(list(data.get(0).values())[0].values())[0].shape[1]
        self.class_num = len(data)
        self.feature_extractor = None
        self.output = None
        self.observe_bus_phase = {}
        self.eval_bus = None
        self.eval_phase = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["data"]
        return state

    def fabricate(self, window: int, shift: int, bus: list, eval_loc: str, **kwargs):
        interval = kwargs.get('interval')
        if interval is None:
            interval = window

        self.feature_extractor = FabricateFeature(window=window, metric_ratio=metric_ratio)

        segment = int((self.total_step - shift) / interval)
        if (self.total_step - shift) % interval:
            segment += 1
        if shift:
            segment += 1

        features = []
        stat_dict = {key: [] for key in data.keys()}

        self.observe_bus_phase = {}
        for x in bus:
            self.observe_bus_phase.update(get_bus_phase(x))

        eval_bus_phase = get_bus_phase(eval_loc)
        self.observe_bus_phase.update(eval_bus_phase)

        assert len(eval_bus_phase) == 1, "Currently support only evaluating at one node."
        for i, phases in eval_bus_phase.items():
            self.eval_bus = i
            assert len(phases) == 1, "Currently support only evaluating at one phase."
            self.eval_phase = phases[0]


        valid_segment = None
        get_feature_names = True
        feature_names = []
        for tap, time_series in data.items():
            valid_segment = 0
            for i in range(1, segment):
                # fabricate features, with loc info, later replace loc with label
                start = min(i * interval + shift, self.total_step)
                end = min(start + interval, self.total_step)
                if start == end:
                    continue

                # feature vector is of window size
                overall_feature = []
                for bus, phase in self.observe_bus_phase.items():
                    phase_list = None
                    if phase == 'all':
                        phase_list = list(time_series[bus].keys())
                    else:
                        assert type(phase) is list, "Phase for extracting features should be in list."
                        phase_list = phase

                    for i in phase_list:
                        phase_v = time_series[bus][i]
                        overall_feature.append(
                            self.feature_extractor.fabricate_feature_vector(
                                phase_v[0, :], phase_v[1, :], start
                            )
                        )
                        if get_feature_names:
                            feature_names.append(
                                [(bus + "." + str(i) + str(fn)) \
                                for fn in self.feature_extractor.feature_names])

                phase_v = time_series[self.eval_bus][self.eval_phase]
                # overall_feature.append(
                #     self.feature_extractor.fabricate_feature_vector(
                #         phase_v[0, :], phase_v[1, :], start, current_tap=tap
                #     )
                # )
                # if get_feature_names:
                #     feature_names.append(
                #         [(self.eval_bus + "." + str(self.eval_phase) + str(fn)) \
                #         for fn in self.feature_extractor.feature_names] + ['tap'])

                get_feature_names = False
                if overall_feature[0] is None:
                    continue
                feature_arr_list = [v for v in overall_feature] + [np.array([tap, valid_segment + self.class_num]).reshape(1, -1)]
                step_feature = np.hstack(tuple(feature_arr_list))
                features.append(step_feature)
                valid_segment += 1

                # evaluation vector is of interval size
                stat_dict[tap].append(self.feature_extractor.get_metric_stat(phase_v[0, :], start, end))

        num_points = len(features)
        features = np.array(features).reshape(num_points, -1)
        feature_names = np.concatenate((np.hstack(feature_names), np.array(['tap', 'label'])))
        feature_pd = pd.DataFrame(data=features, columns=feature_names)

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

        observe_loc = pars['bus']
        eval_loc = pars['eval_loc']

        with open(raw_data_path, "rb") as read:
            data = pickle.load(read)
            fabricator = Fabricator(data, metric_ratio)

            for s in shift:
                fabricator.fabricate(window=window,
                                     shift=s,
                                     bus=observe_loc,
                                     eval_loc=eval_loc,
                                     interval=interval)
                to_save = [fabricator.output, {"window": window, "shift": s, "interval": interval,
                                               "bus_phase": fabricator.observe_bus_phase}]
                with open(data_path + name_generator(window, interval, s, 'pkl', prefix='fabricated'), 'wb') as w:
                    for i in to_save:
                        pickle.dump(i, w)

