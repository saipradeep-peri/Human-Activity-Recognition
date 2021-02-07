# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np

# data_og = pd.read_csv('/content/drive/My Drive/DMProject/cleanData.csv', header=None, names=['timestamp','activityID', 'heart rate'
#                                                                                             ,'IMU_hand_1'
#                                                                                             ,'IMU_hand_2'
#                                                                                             ,'IMU_hand_3'
#                                                                                             ,'IMU_hand_4'
#                                                                                             ,'IMU_hand_5'
#                                                                                             ,'IMU_hand_6'
#                                                                                             ,'IMU_hand_7'
#                                                                                             ,'IMU_hand_8'
#                                                                                             ,'IMU_hand_9'
#                                                                                             ,'IMU_hand_10'
#                                                                                             ,'IMU_hand_11'
#                                                                                             ,'IMU_hand_12'
#                                                                                             ,'IMU_hand_13'
#                                                                                             ,'IMU_hand_14'
#                                                                                             ,'IMU_hand_15'
#                                                                                             ,'IMU_hand_16'
#                                                                                             ,'IMU_hand_17'
#                                                                                             ,'IMU_chest_1'
#                                                                                             ,'IMU_chest_2'
#                                                                                             ,'IMU_chest_3'
#                                                                                             ,'IMU_chest_4'
#                                                                                             ,'IMU_chest_5'
#                                                                                             ,'IMU_chest_6'
#                                                                                             ,'IMU_chest_7'
#                                                                                             ,'IMU_chest_8'
#                                                                                             ,'IMU_chest_9'
#                                                                                             ,'IMU_chest_10'
#                                                                                             ,'IMU_chest_11'
#                                                                                             ,'IMU_chest_12'
#                                                                                             ,'IMU_chest_13'
#                                                                                             ,'IMU_chest_14'
#                                                                                             ,'IMU_chest_15'
#                                                                                             ,'IMU_chest_16'
#                                                                                             ,'IMU_chest_17'
#                                                                                             ,'IMU_ankle_1'
#                                                                                             ,'IMU_ankle_2'
#                                                                                             ,'IMU_ankle_3'
#                                                                                             ,'IMU_ankle_4'
#                                                                                             ,'IMU_ankle_5'
#                                                                                             ,'IMU_ankle_6'
#                                                                                             ,'IMU_ankle_7'
#                                                                                             ,'IMU_ankle_8'
#                                                                                             ,'IMU_ankle_9'
#                                                                                             ,'IMU_ankle_10'
#                                                                                             ,'IMU_ankle_11'
#                                                                                             ,'IMU_ankle_12'
#                                                                                             ,'IMU_ankle_13'
#                                                                                             ,'IMU_ankle_14'
#                                                                                             ,'IMU_ankle_15'
#                                                                                             ,'IMU_ankle_16'
#                                                                                             ,'IMU_ankle_17'
#                                                                                             ,'participant_id'])

# data_og = pd.read_csv('/content/drive/My Drive/DMProject/cleanData_1.csv', index_col=0)

data_og = pd.read_csv('cleanData_1.csv', index_col=0)

data_og.head()

# data_og.to_csv('/content/drive/My Drive/DMProject/cleanData_1.csv')

data_og.shape

data_og['activityID'].value_counts(normalize=True)*100

data_og['heart rate'].value_counts()

data_og['participant_id'].value_counts()

data_og['timestamp'].value_counts()

data_og['IMU_ankle_14'].value_counts()

data_og['IMU_hand_14'].value_counts()

data_og['IMU_chest_14'].value_counts()

data_og.isnull().sum()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

qqplot((data_og['heart rate']), line='s')
pyplot.show()

qqplot((data_og['IMU_hand_1']), line='s')
pyplot.show()

qqplot((data_og['IMU_chest_1']), line='s')
pyplot.show()

qqplot((data_og['IMU_ankle_1']), line='s')
pyplot.show()

qqplot((data_og['IMU_ankle_13']), line='s')
pyplot.show()

qqplot((data_og['IMU_chest_14']), line='s')
pyplot.show()

data_pre_log = data_og.copy()

data_pre_log.head()

data_pre_log = data_pre_log.drop(columns=['timestamp','activityID','IMU_hand_14','IMU_hand_15','IMU_hand_16','IMU_hand_17','IMU_chest_14','IMU_chest_15','IMU_chest_16','IMU_chest_17','IMU_ankle_14','IMU_ankle_15','IMU_ankle_16','IMU_ankle_17','participant_id'])

target_log = data_og[['activityID']]

target_log['activityID'].value_counts()

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
#target_log = pd.DataFrame(minmax_scale.fit_transform(target_log), columns=target_log.columns)
data_pre_log = pd.DataFrame(minmax_scale.fit_transform(data_pre_log), columns=data_pre_log.columns)

data_pre_log.head()

target_log['activityID'] = target_log['activityID'].astype('category').cat.codes

target_log.head()

target_log['activityID'].value_counts(normalize=True)*100

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_pre_log, 
                                                    target_log, 
                                                    test_size=0.3, 
                                                    random_state=1)

from sklearn.naive_bayes import MultinomialNB

log_clf = MultinomialNB()
log_model = log_clf.fit(x_train, y_train)

from sklearn import metrics

y_pred_df = y_test.copy()
y_pred_df['preds'] = log_model.predict(x_test)

prob_df = pd.DataFrame(log_model.predict_proba(x_test))
prob_df.columns = ['prob' + str(col) for col in prob_df.columns]
prob_df.head()

y_pred_df.head()

pd.crosstab(y_pred_df['activityID'], y_pred_df['preds'],
            rownames=['True'], colnames=['Predicted'], margins=True)

metrics.roc_auc_score(y_test, prob_df, multi_class='ovr')

train_prob_df = pd.DataFrame(log_model.predict_proba(x_train))
train_prob_df.columns = ['prob' + str(col) for col in train_prob_df.columns]
train_prob_df.head()

metrics.roc_auc_score(y_train, train_prob_df, multi_class='ovr')

log_model.get_params()

log_model.coef_[0]

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

log_cfm = metrics.confusion_matrix(y_pred_df['activityID'], y_pred_df['preds'])

log_cfm = pd.crosstab(y_pred_df['activityID'], y_pred_df['preds'],
            rownames=['True'], colnames=['Predicted'], margins=True)

plt.figure(figsize = (12,12))
sn.heatmap(log_cfm, annot=True)

print(metrics.classification_report(y_train['activityID'], log_model.predict(x_train)))

print(metrics.classification_report(y_test['activityID'], log_model.predict(x_test)))

"""Random Forest"""

data_pre_rf = data_og.copy()

data_pre_rf = data_pre_rf.drop(columns=['timestamp','activityID','IMU_hand_14','IMU_hand_15','IMU_hand_16','IMU_hand_17','IMU_chest_14','IMU_chest_15','IMU_chest_16','IMU_chest_17','IMU_ankle_14','IMU_ankle_15','IMU_ankle_16','IMU_ankle_17','participant_id'])

target_rf = data_og[['activityID']]

target_rf['activityID'] = target_rf['activityID'].astype('category').cat.codes

target_rf.head()

target_rf['activityID'].value_counts(normalize=True)*100

data_pre_rf.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_pre_rf, 
                                                    target_rf, 
                                                    test_size=0.3, 
                                                    random_state=1)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)

rf_model = rf_clf.fit(x_train, y_train)

rf_train_prob_df = pd.DataFrame(rf_model.predict_proba(x_train))
rf_train_prob_df.columns = ['prob' + str(col) for col in rf_train_prob_df.columns]
rf_train_prob_df.head()

from sklearn import metrics
metrics.roc_auc_score(y_train, rf_train_prob_df, multi_class='ovr')

rf_test_prob_df = pd.DataFrame(rf_model.predict_proba(x_test))
rf_test_prob_df.columns = ['prob' + str(col) for col in rf_test_prob_df.columns]
rf_test_prob_df.head()

metrics.roc_auc_score(y_test, rf_test_prob_df, multi_class='ovr')

pd.crosstab(y_test['activityID'], rf_model.predict(x_test),
            rownames=['True'], colnames=['Predicted'], margins=True)

rf_model.feature_importances_

feat_importances = pd.Series(rf_model.feature_importances_, index=x_train.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh')

print(metrics.classification_report(y_train['activityID'], rf_model.predict(x_train)))

print(metrics.classification_report(y_test['activityID'], rf_model.predict(x_test)))

# pip install shap

import shap

rf_shap_explainer = shap.SamplingExplainer(rf_model.predict_proba, x_train)

rf_shap_vals_train = rf_shap_explainer.shap_values(shap.sample(x_train, 200), approximate=True, nsamples=200)

rf_shap_vals_test = rf_shap_explainer.shap_values(shap.sample(x_test, 200), approximate=True, nsamples=200)

shap.initjs()

shap.summary_plot(rf_shap_vals_train[0], shap.sample(x_train, 200))

shap.summary_plot(rf_shap_vals_test[0], shap.sample(x_test, 200))

shap.dependence_plot('heart rate', rf_shap_vals_train[0], shap.sample(x_train, 200))

rf_shap_vals_train = rf_shap_explainer.shap_values(shap.sample(x_train, 101766), approximate=True, nsamples=200)

"""Light GBM"""

import lightgbm as lgb

data_pre_lgb = data_og.copy()

data_pre_lgb = data_pre_lgb.drop(columns=['timestamp','activityID','IMU_hand_14','IMU_hand_15','IMU_hand_16','IMU_hand_17','IMU_chest_14','IMU_chest_15','IMU_chest_16','IMU_chest_17','IMU_ankle_14','IMU_ankle_15','IMU_ankle_16','IMU_ankle_17','participant_id'])

target_lgb = data_og[['activityID']]

target_lgb['activityID'].value_counts()

target_lgb['activityID'] = target_lgb['activityID'].astype('category').cat.codes

target_lgb['activityID'].value_counts()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_pre_lgb, 
                                                    target_lgb, 
                                                    test_size=0.3, 
                                                    random_state=1)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                    y_train, 
                                                    test_size=0.2, 
                                                    random_state=1)

lgb_data_train = lgb.Dataset(x_train, y_train)
lgb_data_val = lgb.Dataset(x_val, y_val)

lgb_params_mn = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass', # also have multiclassova
    'metric': 'multi_logloss',
    'num_class' : 12,
    'max_depth' : 3,
    #'num_leaves' : ???
    'learning_rate': 0.1,
    #'num_threads' : -1,
    #'scale_pos_weight' : ???
    'early_stopping_round' : 10,
    # min_data_in_leaf = ???,
    # pos_bagging_fraction = ???,
    # neg_bagging_fraction = ???,
    # bagging_freq = ???,
    # max_delta_step = ???,
    #'top_rate' : ???
    #'other_rate' : ???
    #'lambda_l1' : ???
    #'lambda_l2' : ???
}

lgb_gbm_model = lgb.train(params = lgb_params_mn, train_set = lgb_data_train,
                num_boost_round = 100, valid_sets = [lgb_data_val, lgb_data_train],
               valid_names = ['Evaluation', 'Train'])

# a = pd.DataFrame(lgb_gbm_model.predict(x_test, predict_raw_score=True))
# a.columns = ['prob' + str(col) for col in a.columns]
# a.head()

lgb_test_prob_df = pd.DataFrame(lgb_gbm_model.predict(x_test))
lgb_test_prob_df.columns = ['prob' + str(col) for col in lgb_test_prob_df.columns]
lgb_test_prob_df.head()

lgb_test_prob_df.sum(axis=1)

metrics.roc_auc_score(y_test, lgb_test_prob_df, multi_class='ovr')

lgb_test_prob_df = pd.DataFrame(lgb_gbm_model.predict(x_train))
lgb_test_prob_df.columns = ['prob' + str(col) for col in lgb_test_prob_df.columns]
lgb_test_prob_df.head()

metrics.roc_auc_score(y_train, lgb_test_prob_df, multi_class='ovr')

# metrics.confusion_matrix(y_test, lgb_gbm_model.predict(x_test).argmax(axis=1))

pd.crosstab(y_test['activityID'], lgb_gbm_model.predict(x_test).argmax(axis=1),
            rownames=['True'], colnames=['Predicted'], margins=True)

lgb.plot_importance(lgb_gbm_model, max_num_features=10)

print(metrics.classification_report(y_train['activityID'], lgb_gbm_model.predict(x_train).argmax(axis=1)))

print(metrics.classification_report(y_test['activityID'], lgb_gbm_model.predict(x_test).argmax(axis=1)))

