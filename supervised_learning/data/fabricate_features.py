import numpy as np
import pandas as pd
import pickle

# Parameters
window = 10
metric_ratio = 2
v_feature_name = ['v%s' % i for i in range(window)]
c_feature_name = ['c%s' % i for i in range(window)]
angle_feature_name = ['theta%s' % i for i in range(window)]

if __name__ == '__main__':
    raw = 'extracted.pkl'

    with open(raw, "rb") as read:
        data = pickle.load(read)

        total_step = list(data.values())[0].shape[1]
        segment = int(total_step / window) if not total_step % window else \
            int(total_step / window) + 1
        label_num = len(data)
        features = []
        stat_dict = {key: [] for key in data.keys()}

        for tap, time_series in data.items():
            for i in range(1, segment):
                # get metric info in this segment
                # TODO: not possible, can not see other taps here
                # fabricate features, with loc info, later replace loc with label
                pre_start = (i - 1) * window
                start = i * window
                end = min((i + 1) * window, total_step)

                v_vector = np.abs(time_series[0, pre_start: start])
                c_vector = np.abs(time_series[1, pre_start: start])
                angle_vector = np.angle(time_series[0, pre_start: start])
                features.append(np.concatenate((v_vector, c_vector, angle_vector, np.array([tap, i - 1 + label_num]))))

                metric_v = np.abs(time_series[0, start:end])
                stat_dict[tap].append([np.abs(np.mean(metric_v) - 1), np.std(metric_v)])

                # consider from the perspective of using v_vector for 2 use.
        feature_length = len(features[0])
        features = np.array(features)
        feature_pd = pd.DataFrame(data=features,
                                  columns=(v_feature_name + c_feature_name + angle_feature_name + ['tap', 'label']))

        metric = lambda x: x[0] + metric_ratio * x[1]
        for i in range(0, segment-1):
            min_value = 1e6
            for x, stat_series in stat_dict.items():
                score = metric(stat_series[i])
                if score < min_value:
                    min_value = score
                    min_tap = x
            feature_pd["label"].replace({i + label_num: min_tap}, inplace=True)
            # TODO: possible overlap of loc_index and tap_number
            # another way of indexing

        feature_pd.to_csv("fabricated.csv", index=False)