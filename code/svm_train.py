from sklearn.svm import SVC
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

svm = SVC()
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001,'scale','auto'],
              'kernel': ['rbf','poly']}
grid = RandomizedSearchCV(estimator=svm,param_distributions=param_grid,scoring='accuracy',cv=10,n_iter=50)
grid.fit(X_train,y_train)

print('SVM best parameters: ',grid.best_params_)
best_svm= SVC(kernel=grid.best_params_['kernel'],C=grid.best_params_['C'],gamma=grid.best_params_['gamma'],probability=True)
best_svm.fit(X_train,y_train)

scores = cross_validate(best_svm,X_train, y_train,cv=10, scoring='accuracy',return_train_score=True)
print(scores['train_score'])
print(scores['test_score'])

y_pred = best_svm.predict(X_test)
ytrain_pred = best_svm.predict(X_train)

conf_mat = confusion_matrix(data.y[:,0][data.test_mask],y_pred)
cmn = conf_mat.astype('float')/conf_mat.sum(axis=1)[:, np.newaxis]
print('Seagrasses distribution Normalized Confusion Matrix - SVM')
print(cmn)

print(classification_report(data.y[:,0][data.test_mask],y_pred))
df = pd.DataFrame(classification_report(data.y[:,0][data.test_mask],y_pred,output_dict=True)).transpose()
df.to_excel(proj_dir+'\\results\\svm_performances.xlsx')

joblib.dump(best_svm, proj_dir+'\\results\\trained_models\\svm.joblib')