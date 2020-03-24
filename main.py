import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import pickle
# import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--telemetry', help='telemetry data')
    parser.add_argument('--errors', help='errors data')
    parser.add_argument('--maintenance', help='maintenance data')
    parser.add_argument('--failures', help='failures data')
    parser.add_argument('--machines', help='machines data')
    parser.add_argument('--out', help='file to save model to')
    args = parser.parse_args()
    print(args.telemetry)
    telemetry = pd.read_csv(args.telemetry)
    errors = pd.read_csv(args.errors)
    maintenance = pd.read_csv(args.maintenance)
    failures = pd.read_csv(args.failures)
    machines = pd.read_csv(args.machines)
    print("Data loaded")
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

    print('Total number of telemetry records: {}'.format(len(telemetry.index)))
    # print(telemetry.head())
    # print(telemetry.describe())
    # plot(telemetry)

    errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
    errors['errorID'] = errors['errorID'].astype('category')
    print('Total number of error records: {}'.format(len(errors.index)))
    # print(errors.head())

    maintenance['datetime'] = pd.to_datetime(maintenance['datetime'], format="%Y-%m-%d %H:%M:%S")
    maintenance['comp'] = maintenance['comp'].astype('category')
    print('Total number of maintenance records: {}'.format(len(maintenance.index)))
    # print(maintenance.head())

    machines['model'] = machines['model'].astype('category')
    print('Total number of machines: {}'.format(len(machines.index)))
    # print(machines.head())

    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    failures['failure'] = failures['failure'].astype('category')
    print('Total number of failures: {}'.format(len(failures.index)))
    # print(failures.head())

    mean_3h, sd_3h = calc_stats_3h(telemetry)
    mean_24h, sd_24h = calc_stats_24h(telemetry)

    telemetry_feat = pd.concat([mean_3h,
                                sd_3h.iloc[:, 2:6],
                                mean_24h.iloc[:, 2:6],
                                sd_24h.iloc[:, 2:6]], axis=1).dropna()
    # print(telemetry_feat.describe())
    # print(telemetry_feat.head())

    # create a column for each error type
    error_count = pd.get_dummies(errors)
    error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']

    # combine errors for a given machine in a given hour
    error_count = error_count.groupby(['machineID', 'datetime']).sum().reset_index()
    # print(error_count.head(13))

    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'],
                                                             how='left').fillna(0.0)
    # print(error_count.describe())

    temp = []
    fields = ['error%d' % i for i in range(1, 6)]
    for col in fields:
        temp.append(pd.pivot_table(error_count,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).rolling(window=24).sum().resample('3H',
                                                                                 closed='left',
                                                                                 label='right').first().unstack())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    error_count = error_count.dropna()
    # print(error_count.describe())
    # print(error_count.head())

    # create a column for each error type
    comp_rep = pd.get_dummies(maintenance)
    comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

    # combine repairs for a given machine in a given hour
    comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

    # add timepoints where no components were replaced
    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                          on=['datetime', 'machineID'],
                                                          how='outer').fillna(0).sort_values(
        by=['machineID', 'datetime'])

    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        # convert indicator to most recent date of component change
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']

        # forward-fill the most-recent date of component change
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

    # remove dates in 2014 (may have NaN or future component change dates)
    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

    for comp in components:
        comp_rep[comp] = (comp_rep['datetime'] - pd.to_datetime(comp_rep[comp])).apply(lambda x: x / pd.Timedelta(days=1))

    # print(comp_rep.describe())
    # print(error_count.head())
    # print(telemetry_feat.head())

    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')
    # print(final_feat.head())
    print(final_feat.describe())

    labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
    labeled_features = labeled_features.fillna(method='bfill', axis=1, limit=7)  # fill backward up to 24h
    labeled_features = labeled_features.fillna('none')
    # print(labeled_features.head())

    # make test and training splits
    threshold_dates = [[pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
                       [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')],
                       [pd.to_datetime('2015-09-30 01:00:00'), pd.to_datetime('2015-10-01 01:00:00')]]

    test_results = []
    models = []
    for last_train_date, first_test_date in threshold_dates:
        # split out training and test data
        train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
        train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                            'machineID',
                                                                                                            'failure'],
                                                                                                           1))
        test_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] > first_test_date].drop(['datetime',
                                                                                                           'machineID',
                                                                                                           'failure'],
                                                                                                          1))

        # train and predict using the model, storing results for later
        my_model = GradientBoostingClassifier(random_state=42)
        my_model.fit(train_X, train_y)
        test_result = pd.DataFrame(labeled_features.loc[labeled_features['datetime'] > first_test_date])
        test_result['predicted_failure'] = my_model.predict(test_X)
        test_results.append(test_result)
        models.append(my_model)

    pickle.dump(models, open(args.out, mode='wb+'))


# def plot(telemetry):
#     plot_df = telemetry.loc[(telemetry['machineID'] == 1) &
#                             (telemetry['datetime'] > pd.to_datetime('2015-01-01')) &
#                             (telemetry['datetime'] < pd.to_datetime('2015-02-01')), ['datetime', 'volt']]
#
#     sns.set_style("darkgrid")
#     plt.figure(figsize=(12, 6))
#     plt.plot(plot_df['datetime'], plot_df['volt'])
#     plt.ylabel('voltage')
#
#     # make x-axis ticks legible
#     adf = plt.gca().get_xaxis().get_major_formatter()
#     adf.scaled[1.0] = '%m-%d'
#     plt.xlabel('Date')
#     plt.show()


def calc_stats_3h(telemetry):
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).resample('3H', closed='left', label='right').mean().unstack())
    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)

    # repeat for standard deviation
    temp = []
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).resample('3H', closed='left', label='right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)

    return telemetry_mean_3h, telemetry_sd_3h


def calc_stats_24h(telemetry):
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).rolling(window=24).mean().resample('3H',
                                                                                  closed='left',
                                                                                  label='right').first().unstack())
    telemetry_mean_24h = pd.concat(temp, axis=1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

    # repeat for standard deviation
    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                   index='datetime',
                                   columns='machineID',
                                   values=col).rolling(window=24).std().resample('3H',
                                                                                 closed='left',
                                                                                 label='right').first().unstack())
    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
    telemetry_sd_24h.reset_index(inplace=True)
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]

    # Notice that a 24h rolling average is not available at the earliest timepoints
    return telemetry_mean_24h, telemetry_sd_24h

#
# def Evaluate(predicted, actual, labels):
#     output_labels = []
#     output = []
#
#     # Calculate and display confusion matrix
#     cm = confusion_matrix(actual, predicted, labels=labels)
#     print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
#     print(cm)
#
#     # Calculate precision, recall, and F1 score
#     accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
#     precision = precision_score(actual, predicted, average=None, labels=labels)
#     recall = recall_score(actual, predicted, average=None, labels=labels)
#     f1 = 2 * precision * recall / (precision + recall)
#     output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
#     output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])
#
#     # Calculate the macro versions of these metrics
#     output.extend([[np.mean(precision)] * len(labels),
#                    [np.mean(recall)] * len(labels),
#                    [np.mean(f1)] * len(labels)])
#     output_labels.extend(['macro precision', 'macro recall', 'macro F1'])
#
#     # Find the one-vs.-all confusion matrix
#     cm_row_sums = cm.sum(axis=1)
#     cm_col_sums = cm.sum(axis=0)
#     s = np.zeros((2, 2))
#     for i in range(len(labels)):
#         v = np.array([[cm[i, i],
#                        cm_row_sums[i] - cm[i, i]],
#                       [cm_col_sums[i] - cm[i, i],
#                        np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
#         s += v
#     s_row_sums = s.sum(axis=1)
#
#     # Add average accuracy and micro-averaged  precision/recall/F1
#     avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
#     micro_prf = [float(s[0, 0]) / s_row_sums[0]] * len(labels)
#     output.extend([avg_accuracy, micro_prf])
#     output_labels.extend(['average accuracy',
#                           'micro-averaged precision/recall/F1'])
#
#     # Compute metrics for the majority classifier
#     mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
#     cm_row_dist = cm_row_sums / float(np.sum(cm))
#     mc_accuracy = 0 * cm_row_dist;
#     mc_accuracy[mc_index] = cm_row_dist[mc_index]
#     mc_recall = 0 * cm_row_dist;
#     mc_recall[mc_index] = 1
#     mc_precision = 0 * cm_row_dist
#     mc_precision[mc_index] = cm_row_dist[mc_index]
#     mc_F1 = 0 * cm_row_dist;
#     mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
#     output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
#                    mc_precision.tolist(), mc_F1.tolist()])
#     output_labels.extend(['majority class accuracy', 'majority class recall',
#                           'majority class precision', 'majority class F1'])
#
#     # Random accuracy and kappa
#     cm_col_dist = cm_col_sums / float(np.sum(cm))
#     exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
#     kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
#     output.extend([exp_accuracy.tolist(), kappa.tolist()])
#     output_labels.extend(['expected accuracy', 'kappa'])
#
#     # Random guess
#     rg_accuracy = np.ones(len(labels)) / float(len(labels))
#     rg_precision = cm_row_dist
#     rg_recall = np.ones(len(labels)) / float(len(labels))
#     rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
#     output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
#                    rg_recall.tolist(), rg_F1.tolist()])
#     output_labels.extend(['random guess accuracy', 'random guess precision',
#                           'random guess recall', 'random guess F1'])
#
#     # Random weighted guess
#     rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist ** 2)
#     rwg_precision = cm_row_dist
#     rwg_recall = cm_row_dist
#     rwg_F1 = cm_row_dist
#     output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
#                    rwg_recall.tolist(), rwg_F1.tolist()])
#     output_labels.extend(['random weighted guess accuracy',
#                           'random weighted guess precision',
#                           'random weighted guess recall',
#                           'random weighted guess F1'])
#
#     output_df = pd.DataFrame(output, columns=labels)
#     output_df.index = output_labels
#
#     return output_df


if __name__ == '__main__':
    main()
