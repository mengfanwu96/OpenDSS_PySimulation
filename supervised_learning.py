import json
from src.supervised_learning.supervised_learning_flow import SupervisedLearning

# load parameters
with open("supervised_learning.json") as par:
    pars = json.load(par)
    root = pars['root']
    cross_validation = pars['cross_validation']
    data_path = pars['data_dir']
    window = pars['window']
    interval = pars['interval']  # different interval might create same sample in train and test data
    train_shift = pars['train_shift']
    feature_selection_limit = pars['feature_selection_number']
    fs_standard = pars['feature_selection_criteria']

    label_name = 'label'

    flow = SupervisedLearning(cross_validation, root, data_path, window, train_shift, label_name,
                              feature_selection_criteria=fs_standard,
                              feature_selection_number=feature_selection_limit)

    if cross_validation:
        flow.cross_validation(pars.get('fold_number'))

    flow.train()
    flow.test(pars.get('test_shift'))
    flow.save_model(root + pars.get('model_dir'))