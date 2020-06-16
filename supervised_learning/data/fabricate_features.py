import numpy as np
import pandas as pd
import pickle


# Parameters
class fabricate_feature:
    def __init__(self, window: int = 10, metric_ratio=2.0):
        self.window = window
        self.metric_ratio = metric_ratio
        self.v_feature_name = ['v%s' % i for i in range(window)]
        self.c_feature_name = ['c%s' % i for i in range(window)]
        self.angle_feature_name = ['theta%s' % i for i in range(window)]
        self.feature_names = self.v_feature_name + self.c_feature_name + \
                             self.angle_feature_name + ['tap']
        self.metric = [lambda y: np.mean(y) - 1, np.std]
        self.metric_ratio = metric_ratio
        # TODO: maybe other complicated metric calculation

    def fabricate_feature_vector(self, v, c, final_step:int, current_tap: int):
        v_vector = np.abs(v[final_step - self.window: final_step])
        c_vector = np.abs(c[final_step - self.window: final_step])
        angle_vector = np.angle(v[final_step - self.window: final_step])

        return np.concatenate((v_vector, c_vector, angle_vector, np.array(current_tap)))

    def get_metric_stat(self, v, final_step:int):
        return [f(v[final_step - self.window: final_step]) for f in self.metric]

    def evaluate_metric(self, metric_stat:np.array):
        if type(self.metric_ratio) is float and len(metric_stat) == 2:
            return metric_stat[0] + self.metric_ratio * metric_stat[1]
        elif type(self.metric_ratio) is list and len(self.metric_ratio) == len(metric_stat) - 1:
            a = metric_stat[0]
            for idx, k in enumerate(self.metric_ratio):
                a += metric_stat[idx + 1] * k
            return a
        else:
            assert False, "Error in computing metrics, ratios: %s and statistics: %s" % (self.metric_ratio, metric_stat)

if __name__ == '__main__':
    raw = 'extracted.pkl'

    fv = fabricate_feature(window=10, metric_ratio=2)
    shift = 0 % fv.window

    with open(raw, "rb") as read:
        data = pickle.load(read)

        total_step = list(data.values())[0].shape[1]
        segment = int(total_step / fv.window) if not total_step % fv.window else \
            int(total_step / fv.window) + 1
        label_num = len(data)
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
                features.append(np.concatenate((feature_v, np.array(i - 1 + label_num))))

                metric_v = np.abs(time_series[0, start:end])
                stat_dict[tap].append(fv.get_metric_stat(time_series[0, :], end))

        feature_length = len(features[0])
        features = np.array(features)
        feature_pd = pd.DataFrame(data=features,
                                  columns=(fv.feature_names + ['label']))

        for i in range(0, segment-1):
            min_value = 1e6
            for x, stat_series in stat_dict.items():
                score = fv.evaluate_metric(stat_series[i])
                if score < min_value:
                    min_value = score
                    min_tap = x
            feature_pd["label"].replace({i + label_num: min_tap}, inplace=True)
            # possible overlap of loc_index and tap_number
            # another way of indexing

            # TODO: test if still working in fabricating

        feature_pd.to_csv("fabricated_w%s_d%s.csv" % (fv.window, shift), index=False)
        #  TODO: also include how to extract features
