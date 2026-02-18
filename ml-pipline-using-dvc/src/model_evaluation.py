import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

clf = pickle.load(open('models/model.pkl','rb'))
X_test = sp.load_npz('data/features/X_test.npz')
y_test = pd.read_csv('data/features/y_test.csv')

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)