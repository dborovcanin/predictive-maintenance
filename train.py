import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import numpy as np
import pickle
# import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='data')
    parser.add_argument('--out', help='model location')
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset)
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], format="%Y-%m-%d %H:%M:%S")
    print("Data loaded")

    print('Total number of telemetry records: {}'.format(len(dataset.index)))
    print(dataset.head())
    print(dataset.describe())

    # make test and training splits
    threshold_dates = [[pd.to_datetime('2015-08-15 01:00:00'), pd.to_datetime('2015-08-26 01:00:00')],
                       [pd.to_datetime('2015-08-16 01:00:00'), pd.to_datetime('2015-09-05 01:00:00')],
                       [pd.to_datetime('2015-09-06 01:00:00'), pd.to_datetime('2015-09-18 01:00:00')]]

    test_results = []
    models = []
    for last_train_date, first_test_date in threshold_dates:
        # split out training and test data
        print(last_train_date, first_test_date)
        train_y = dataset.loc[dataset['datetime'] < last_train_date, 'failure']
        train_X = pd.get_dummies(dataset.loc[dataset['datetime'] < last_train_date].drop(['datetime',
                                                                                                            'machineID',
                                                                                                            'failure'],
                                                                                                           1))
        test_X = pd.get_dummies(dataset.loc[dataset['datetime'] > first_test_date].drop(['datetime',
                                                                                                           'machineID',
                                                                                                           'failure'],
                                                                                                          1))
        print("TRESHOLD")
        print(len(train_X.index))
        print(len(train_y.index))
        print(len(test_X.index))

        # train and predict using the model, storing results for later
        my_model = GradientBoostingClassifier(random_state=42)
        my_model.fit(train_X, train_y)
        test_result = pd.DataFrame(dataset.loc[dataset['datetime'] > first_test_date])
        test_result['predicted_failure'] = my_model.predict(test_X)
        test_results.append(test_result)
        models.append(my_model)

    print('saving model...')
    pickle.dump(models, open(args.out, mode='wb+'))

    evaluation_results = []
    for i, test_result in enumerate(test_results):
        print('\nSplit %d:' % (i + 1))
        evaluation_result = evaluate(actual=test_result['failure'],
                                     predicted=test_result['predicted_failure'],
                                     labels=['none', 'comp1', 'comp2', 'comp3', 'comp4'])
        evaluation_results.append(evaluation_result)
    print(evaluation_results[0])  # show full results for first split only
    recall_df = pd.DataFrame([evaluation_results[0].loc['recall'].values,
                              evaluation_results[1].loc['recall'].values,
                              evaluation_results[2].loc['recall'].values],
                             columns=['none', 'comp1', 'comp2', 'comp3', 'comp4'],
                             index=['recall for first split',
                                    'recall for second split',
                                    'recall for third split'])
    print('\n\n', recall_df)


def evaluate(predicted, actual, labels):
    output_labels = []
    output = []

    # Calculate and display confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
    print(cm)

    # Calculate precision, recall, and F1 score
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    precision = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])

    # Calculate the macro versions of these metrics
    output.extend([[np.mean(precision)] * len(labels),
                   [np.mean(recall)] * len(labels),
                   [np.mean(f1)] * len(labels)])
    output_labels.extend(['macro precision', 'macro recall', 'macro F1'])

    # Find the one-vs.-all confusion matrix
    cm_row_sums = cm.sum(axis=1)
    cm_col_sums = cm.sum(axis=0)
    s = np.zeros((2, 2))
    for i in range(len(labels)):
        v = np.array([[cm[i, i],
                       cm_row_sums[i] - cm[i, i]],
                      [cm_col_sums[i] - cm[i, i],
                       np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
        s += v
    s_row_sums = s.sum(axis=1)

    # Add average accuracy and micro-averaged  precision/recall/F1
    avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
    micro_prf = [float(s[0, 0]) / s_row_sums[0]] * len(labels)
    output.extend([avg_accuracy, micro_prf])
    output_labels.extend(['average accuracy',
                          'micro-averaged precision/recall/F1'])

    # Compute metrics for the majority classifier
    mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
    cm_row_dist = cm_row_sums / float(np.sum(cm))
    mc_accuracy = 0 * cm_row_dist;
    mc_accuracy[mc_index] = cm_row_dist[mc_index]
    mc_recall = 0 * cm_row_dist;
    mc_recall[mc_index] = 1
    mc_precision = 0 * cm_row_dist
    mc_precision[mc_index] = cm_row_dist[mc_index]
    mc_F1 = 0 * cm_row_dist;
    mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
    output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
                   mc_precision.tolist(), mc_F1.tolist()])
    output_labels.extend(['majority class accuracy', 'majority class recall',
                          'majority class precision', 'majority class F1'])

    # Random accuracy and kappa
    cm_col_dist = cm_col_sums / float(np.sum(cm))
    exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
    kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
    output.extend([exp_accuracy.tolist(), kappa.tolist()])
    output_labels.extend(['expected accuracy', 'kappa'])

    # Random guess
    rg_accuracy = np.ones(len(labels)) / float(len(labels))
    rg_precision = cm_row_dist
    rg_recall = np.ones(len(labels)) / float(len(labels))
    rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
    output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
                   rg_recall.tolist(), rg_F1.tolist()])
    output_labels.extend(['random guess accuracy', 'random guess precision',
                          'random guess recall', 'random guess F1'])

    # Random weighted guess
    rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist ** 2)
    rwg_precision = cm_row_dist
    rwg_recall = cm_row_dist
    rwg_F1 = cm_row_dist
    output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
                   rwg_recall.tolist(), rwg_F1.tolist()])
    output_labels.extend(['random weighted guess accuracy',
                          'random weighted guess precision',
                          'random weighted guess recall',
                          'random weighted guess F1'])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels

    return output_df


if __name__ == '__main__':
    main()
