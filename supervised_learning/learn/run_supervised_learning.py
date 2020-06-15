import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter


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
data = pd.read_csv('./data/fabricated.csv')
columns = data.columns
features_names = list(columns)
features_names.remove(label_name)

features = data[features_names]
label = np.array(data[label_name])

# pre-processing
scaler = preprocessing.MinMaxScaler()
scaler.fit(np.array(features))
features_np = scaler.transform(np.array(features))

# learn
model = MLPClassifier(solver='adam', hidden_layer_sizes=(300, 300, 300), alpha=0.01, activation='relu', max_iter=400)
kf = KFold(n_splits=8, shuffle=True, random_state=31337)

err = []
extreme_result = []
for train_index, test_index in kf.split(features_np):
    x_train = features_np[train_index]
    y_train = label[train_index]
    x_test = features_np[test_index]
    y_test = label[test_index]

    selector = SelectKBest(mutual_info_classif, k=15)

    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    model.fit(x_train_selected, y_train)
    predictions = model.predict(x_test_selected)

    extreme_result.append(diff_count_extreme(y_test, predictions))
    diff_count(dict(Counter(extreme_result[-1][:, 1])))

    err.append(accuracy_score(y_test, predictions))
    # diff_count(dict(Counter(y_test - predictions)))

print("error is %s , deviation is %s" % (np.mean(err), np.std(err)))



