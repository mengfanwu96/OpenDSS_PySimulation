import json
import pickle
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, f_regression, mutual_info_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from .fabricating.fabricate_features import Fabricator, FabricateFeature, name_generator


def get_data(window, interval, shift, path, prefix='fabricated', extract_par=False):
    # first object is the pd DataFrame
    # second object is the parameter info
    name = name_generator(window, interval, shift, 'pkl', prefix)
    with open(path + name, 'rb') as r:
        data = pickle.load(r)
        if extract_par:
            data = tuple([data, pickle.load(r)])
    return data


def diff_count(b: dict):
    a = np.zeros(32)

    for diff, num in b.items():
        abs_diff = int(diff if diff >= 0 else -diff)
        a[abs_diff] += num

    for i in range(len(a)):
        print("Difference: %s, count: %s" % (i, a[i]))

    print("Accepted tap difference counter: %s out of %s" % (np.sum(a[0:4]), np.sum(list(b.values()))))
    # TODO: another metric to really calculate the ratio of two taps


def diff_count_extreme(label, predictions):
    if type(label) is not np.ndarray:
        label = np.array(label)
    assert len(label) == len(predictions)

    res = []
    for i in range(len(label)):
        if np.abs(label[i]) >= 5:
            res.append([label[i], label[i] - predictions[i]])

    return np.array(res)


def get_feature_names(data: pd.DataFrame, label_name):
    feature_names = list(data.columns)
    feature_names.remove(label_name)
    return feature_names


def evaluation_print(y_test, predictions):
    extreme_result = diff_count_extreme(np.array(y_test), predictions)
    print("For optimal taps greater than 5:")
    diff_count(dict(Counter(extreme_result[:, 1])))

    print("For all taps:")
    diff_count(dict(Counter(np.array(y_test) - predictions)))


def to_integer_tap(predictions, limit=16):
    res = np.array(predictions)
    res_int = np.around(predictions).astype(int)
    ex_pos_loc = np.where(res_int > limit)
    ex_neg_loc = np.where(res_int < -limit)
    res_int[ex_pos_loc] = limit
    res_int[ex_neg_loc] = -limit

    return res_int


feature_selection_method = {
    'chi': chi2,
    'f_c': f_classif,
    'f_r': f_regression,
    'mutual_info_c': mutual_info_classif,
    'mutual_info_r': mutual_info_regression,
}


regression_model = {
    'MLP': MLPRegressor,
}


class SupervisedLearning:
    def __init__(self, cv, root: str, data_path: str, window, train_shift: list, label_name="label", **kwargs):
        self.cv = cv
        self.root = root
        self.data_path = root + data_path
        self.label_name = label_name
        self.feature_names = None
        self.window = window
        self.interval = window if kwargs.get('interval') is None else kwargs.get('interval')
        self.train_shift = train_shift
        self.feature_sel_num = kwargs.get("feature_selection_number")
        self.feature_sel_method = feature_selection_method.get(kwargs.get("feature_selection_criteria"))
        if self.feature_sel_method is None:
            self.feature_sel_method = f_regression

        self.model = regression_model.get(kwargs.get('predictor'))
        if self.model is None:
            self.model = MLPRegressor(solver='adam', hidden_layer_sizes=(300, 300, 300),
                                                        alpha=0.01, activation='relu', max_iter=400)

        self.pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
                                    ('feature_selector', SelectKBest(self.feature_sel_method, k=self.feature_sel_num)),
                                    ('predictor', self.model)])
        self.feature_fabrication_pars = None

    def cross_validation(self, k=8):
        data, self.feature_fabrication_pars = get_data(self.window, self.interval,
                                                       self.train_shift[0], self.data_path,
                                                       extract_par=True)

        self.feature_names = get_feature_names(data, self.label_name)
        features = data[self.feature_names]
        label = data[self.label_name]

        kf = KFold(n_splits=k, shuffle=True)
        err = []
        for train_index, test_index in kf.split(features):
            x_train = features.iloc[train_index]
            y_train = label.iloc[train_index]
            x_test = features.iloc[test_index]
            y_test = label.iloc[test_index]

            self.pipe.fit(X=x_train, y=y_train)
            print("Selected feature indices: % s" % self.pipe['feature_selector'].get_support(indices=True))

            predictions = self.pipe.predict(x_test)
            predictions = to_integer_tap(predictions)

            evaluation_print(y_test, predictions)
            err.append(accuracy_score(np.array(y_test), predictions))

        print("Average accuracy score is %s , deviation is %s" % (np.mean(err), np.std(err)))

    def train(self):
        train_data = None
        for idx, shift in enumerate(self.train_shift):
            # data = pd.read_csv(data_path + "fabricated_w%s_int%s_d%s.csv" % (window, interval, shift))
            data, self.feature_fabrication_pars = get_data(self.window, self.interval,
                                                          shift, self.data_path, extract_par=True)
            if idx:
                train_data.append(data)
            else:
                train_data = data

        self.feature_names = get_feature_names(train_data, self.label_name)
        train_features = train_data[self.feature_names]
        train_label = train_data[self.label_name]

        print("Points in training set: %s" % list(train_features.shape))
        self.pipe.fit(train_features, train_label)
        print("Model fitted.")
        print("Selected feature indices: % s" % self.pipe['feature_selector'].get_support(indices=True))

    def save_model(self, path):
        shift = self.train_shift[0] if self.cv else self.train_shift
        name = name_generator(self.window, self.interval, shift, 'pkl')

        path_saving = path + name
        with open(path_saving, 'wb') as w:
            for m in [self.pipe, self.feature_fabrication_pars]:
                pickle.dump(m, w)

    def test(self, test_shift: list):
        test_data = None
        for idx, shift in enumerate(test_shift):
            data = get_data(self.window, self.interval, shift, self.data_path)
            if idx:
                test_data.append(data)
            else:
                test_data = data

        test_features = test_data[self.feature_names]
        test_label = test_data[self.label_name]

        predictions = self.pipe.predict(test_features)
        predictions = to_integer_tap(predictions)

        evaluation_print(test_label, predictions)
        print("Accuracy score is: " + str(accuracy_score(np.array(test_label), predictions)))
