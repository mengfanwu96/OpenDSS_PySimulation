import numpy as np
import pandas as pd
import pickle


# Parameters
window = 10
metric_ratio = 2

if __name__ == '__main__':
    raw = 'extracted.pkl'

    with open(raw, "rb") as read:
        data = pickle.load(read)
        a = 1

        total_step = data.values[0].shape[1]
        segment = int(total_step / window) if not total_step % window else \
            int(total_step / window) + 1
        label_num = len(data)
        features = []
        stat_dict = {}

        for tap, time_series in data.items():
            total_step = time_series.shape[1]


            for i in range(1, segment):
                # get metric info in this segment
                # TODO: not possible, can not see other taps here
                # fabricate features, with loc info, later replace loc with label
                pre_start = (i - 1) * window
                start = i * window
                end = min((i + 1) * window, total_step)

                v_vector = np.abs(time_series[0, pre_start: start])
                c_vector = np.abs(time_series[1, pre_start: start])
                features.append(np.concatenate((v_vector, c_vector, np.array(tap, start)), axis=1))
                stat_dict[tap]

                # consider from the perspective of using v_vector for 2 use.

                # remember that the tap of last segment is also part of the feature
