import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from collections import Counter
import joblib

root = 'C:/Users/chmewu/OpenDSS_PySimulation/supervised_learning/'
cross_validation = False


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
    assert len(label) == len(predictions)

    res = []
    for i in range(len(label)):
        if np.abs(label[i]) >= 5:
            res.append([label[i], label[i] - predictions[i]])

    return np.array(res)


# data info
window = 10
metric_ratio = 2
v_feature_name = ['v%s' % i for i in range(window)]
c_feature_name = ['c%s' % i for i in range(window)]
angle_feature_name = ['theta%s' % i for i in range(window)]
label_name = 'label'

# get data
data = pd.read_csv(root + '/data/fabricated.csv')
columns = data.columns
features_names = list(columns)
features_names.remove(label_name)
feature_num = len(features_names)

# pre-processing
pipe = Pipeline(steps=[('scaler', MinMaxScaler()),
                       ('feature_selector', SelectKBest(mutual_info_classif, k=15)),
                       ('classifier', MLPClassifier(solver='adam', hidden_layer_sizes=(300, 300, 300),
                                                    alpha=0.01, activation='relu', max_iter=400))])

features = data[features_names]
label = data[label_name]

if cross_validation:
    kf = KFold(n_splits=8, shuffle=True)

    err = []
    extreme_result = []
    for train_index, test_index in kf.split(features):
        x_train = features.iloc[train_index]
        y_train = label.iloc[train_index]
        x_test = features.iloc[test_index]
        y_test = label.iloc[test_index]

        pipe.fit(X=x_train, y=y_train)

        print("Selected feature indices: % s" % pipe['feature_selector'].get_support(indices=True))

        predictions = pipe.predict(x_test)

        extreme_result.append(diff_count_extreme(np.array(y_test), predictions))
        print("For optimal taps greater than 5:")
        diff_count(dict(Counter(extreme_result[-1][:, 1])))

        err.append(accuracy_score(np.array(y_test), predictions))
        diff_count(dict(Counter(np.array(y_test) - predictions)))

    print("Average accuracy score is %s , deviation is %s" % (np.mean(err), np.std(err)))
else:
    pipe.fit(features, label)
    path_saving = root + '/model/shift_.pkl'   # TODO: read the name from csv (need a new name from fabricating)
    # TODO: how to store feature selection?

    joblib.dump(pipe, path_saving)





