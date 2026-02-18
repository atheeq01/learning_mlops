import os
import joblib
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

train_data = sp.load_npz('data/features/X_train.npz')
train_labels = pd.read_csv('data/features/y_train.csv')

xg_model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=1000,n_jobs=-1)
xg_model.fit(train_data, train_labels)

print("Saving model...")
os.makedirs("models", exist_ok=True)
joblib.dump(xg_model, "models/model.pkl")

