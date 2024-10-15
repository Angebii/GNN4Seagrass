from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import torch
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import sys
import joblib

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
data=torch.load(proj_dir+'\\data\\dataset.pth')

X_training = data.x[data.train_mask].detach().numpy()
y_training = data.y[data.train_mask].detach().numpy()

X_val = data.x[data.val_mask].detach().numpy()
y_val = data.y[data.val_mask].detach().numpy()

X_test = data.x[data.test_mask].detach().numpy()
y_test = data.y[data.test_mask].detach().numpy()

X_train=np.concatenate((X_training,X_val),axis=0)
y_train=np.concatenate((y_training,y_val),axis=0)
y_train = np.squeeze(y_train)

# Number of trees in random forest
n_estimators = [30,50,70,100,120]

# Maximum number of levels in tree
max_depth = [5,7,None]

# Minimum number of samples required to split a node
min_samples_split = [2,3, 5,7]

min_samples_leaf = [1, 2]

class_weight = [None,'balanced']

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'class_weight': class_weight
               }

# Use the random grid to search for best hyperparameters
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(rf, random_grid, n_iter=15, cv = 10, verbose = 4, n_jobs = -1,scoring='accuracy')
# Fit the random search model
rf_random.fit(X_train, y_train)

print('Random Forest best parameters: ', rf_random.best_params_)

best_rf = RandomForestClassifier(n_estimators=rf_random.best_params_['n_estimators'], min_samples_split=rf_random.best_params_['min_samples_split'], min_samples_leaf=rf_random.best_params_['min_samples_leaf'], max_depth=rf_random.best_params_['max_depth'] ,class_weight=rf_random.best_params_['class_weight'])
best_rf.fit(X_train, y_train)

scores = cross_validate(best_rf,X_train, y_train,cv=10, scoring='accuracy',return_train_score=True)
print(scores['train_score'])
print(scores['test_score'])

y_pred = best_rf.predict(X_test)
ytrain_pred = best_rf.predict(X_train)

conf_mat = confusion_matrix(data.y[:,0][data.test_mask],y_pred)
cmn = conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
print('Seagrasses distribution Normalized Confusion Matrix - RF')
print(cmn)

print(classification_report(data.y[:,0][data.test_mask],y_pred))
df = pd.DataFrame(classification_report(data.y[:,0][data.test_mask],y_pred,output_dict=True)).transpose()
df.to_excel(proj_dir+'\\results\\rf_performances.xlsx')

joblib.dump(best_rf, proj_dir+'\\results\\trained_models\\rf.joblib')