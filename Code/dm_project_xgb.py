# -*- coding: utf-8 -*-



import pandas as pd

origDf = pd.read_csv('/content/drive/My Drive/DMProject/cleanData_1.csv',index_col = 0)
origDf.head()

origDf['activityID'].value_counts(normalize = True)*100

targetColOrig = origDf['activityID']
targetCol = targetColOrig.astype('category').cat.codes
targetCol

origDf = origDf.drop(columns=['timestamp','activityID','IMU_hand_14','IMU_hand_15','IMU_hand_16','IMU_hand_17','IMU_chest_14','IMU_chest_15','IMU_chest_16','IMU_chest_17','IMU_ankle_14','IMU_ankle_15','IMU_ankle_16','IMU_ankle_17','participant_id'])
origDf.head()

import xgboost as xgb

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(origDf, 
                                                    targetCol, 
                                                    test_size=0.3, 
                                                    random_state=42)

dtrain = xgb.DMatrix(data = x_train, label = y_train)
dval = xgb.DMatrix(data = x_val, label = y_val)

param = {'max_depth': 4,
         'eta': 0.3,
         'silent':1,
         'objective':'multi:softmax',
         'eval_metric': 'merror',
         #'scale_pos_weight' : 0.5,
         'maximize' : 'FALSE',
         'n_jobs' : 20,
         'tree_method': 'gpu_hist',
         'num_class': 12
        }

watchlist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 300
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=10)

from sklearn import metrics

from sklearn import metrics
#scores_test = mort_test_w_preds['xgb_probs']
#scores_train = mort_train_w_preds['xgb_probs']
# fpr, tpr, thresholds = metrics.roc_curve(y_train, scores_train,)
# metrics.auc(fpr, tpr)
metrics.roc_auc_score(y_train,bst.predict(dtrain),multi_class='ovr')

from sklearn import metrics
#scores_test = mort_test_w_preds['xgb_probs']
#scores_train = mort_train_w_preds['xgb_probs']
# fpr, tpr, thresholds = metrics.roc_curve(y_train, scores_train,)
# metrics.auc(fpr, tpr)
metrics.roc_auc_score(y_val,bst.predict(dval),multi_class='ovr')

bst.best_iteration

from xgboost import plot_importance
plot_importance(bst,max_num_features=10,importance_type='gain',title='XGBoost Feature Importance')

bst.predict(dval)

pd.crosstab(y_val, bst.predict(dval),
            rownames=['True'], colnames=['Predicted'], margins=True)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Classifiaction_report : ")
print(classification_report( y_train, bst.predict(dtrain),digits=10))

print("Classifiaction_report : ")
print(classification_report( y_val, bst.predict(dval),digits=10))

origDf.head()

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
#target_log = pd.DataFrame(minmax_scale.fit_transform(target_log), columns=target_log.columns)
data_pre_log = pd.DataFrame(minmax_scale.fit_transform(origDf), columns= origDf.columns)

data_pre_log.head()

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
# data_transformed['clust_grp'] = clusterer.predict(data_pre_log)

from sklearn import metrics
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_pre_log)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

from sklearn.cluster import KMeans
clusterer = KMeans(5, random_state=1)
clusterer.fit(data_pre_log)
# Predict values
data_pre_log['clust_grp'] = clusterer.predict(data_pre_log)

data_pre_log.head()

data_pre_log['activityId'] = targetColOrig
data_pre_log.head()

type(data_pre_log[['heart rate','IMU_chest_7','IMU_chest_5','clust_grp']])

data_pre_log[['heart rate','IMU_chest_7','IMU_chest_4','clust_grp']].to_csv('/content/data_forClustering.csv')

import plotly.express as px
fig = px.scatter(data_pre_log.head(500000), x="IMU_chest_7", y="activityId", color="clust_grp")
fig.show()

clusterer.cluster_centers_

from mpl_toolkits import mplot3d

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

import plotly.express as px
fig = px.scatter_3d(data_pre_log.head(500000), x='heart rate', y='IMU_chest_7', z='IMU_chest_4',
              color='clust_grp')
fig.show()

newDf = origDf[['heart rate','activityID','IMU_chest_7','IMU_chest_4']].copy()

newDf.head()

newDf1 = newDf[(newDf['activityID']==6) | (newDf['activityID']==5) | (newDf['activityID']==3) | (newDf['activityID']==4)]
newDf1.head()

newDf1.shape

targetColOrig = newDf['activityID']
targetCol = targetColOrig.astype('category').cat.codes
targetCol

newDf1 = newDf1.drop(columns = ['activityID'])
newDf1.head()

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
#target_log = pd.DataFrame(minmax_scale.fit_transform(target_log), columns=target_log.columns)
newDf1 = pd.DataFrame(minmax_scale.fit_transform(newDf1), columns= newDf1.columns)

newDf1.head()

newDf = newDf.drop(columns = ['activityID'])
newDf.head()

x_train, x_val, y_train, y_val = train_test_split(newDf, 
                                                    targetCol, 
                                                    test_size=0.4, 
                                                    random_state=42)

from sklearn.svm import SVC 
svm_model_rbf = SVC(kernel = 'rbf', C = 1).fit(x_train, y_train)

svm_model_rbf

svm_predictions = svm_model_rbf.predict(x_val)

svm_predictions

# model accuracy for X_test   
accuracy = svm_model_rbf.score(x_val, y_val)

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions)

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

params_grid

dtrain = xgb.DMatrix(data = x_train, label = y_train)
dval = xgb.DMatrix(data = x_val, label = y_val)

param = {'max_depth': 5,
         'eta': 0.3,
         'silent':1,
         'objective':'multi:softprob',
         'eval_metric': 'merror',
         #'scale_pos_weight' : 0.5,
         'maximize' : 'FALSE',
         'n_jobs' : 20,
         'tree_method': 'gpu_hist',
         'num_class': 12
        }

watchlist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 300
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=10)

from sklearn import metrics
metrics.roc_auc_score(y_train,bst.predict(dtrain),multi_class='ovr')

from sklearn import metrics
#scores_test = mort_test_w_preds['xgb_probs']
#scores_train = mort_train_w_preds['xgb_probs']
# fpr, tpr, thresholds = metrics.roc_curve(y_train, scores_train,)
# metrics.auc(fpr, tpr)
metrics.roc_auc_score(y_val,bst.predict(dval),multi_class='ovr')

