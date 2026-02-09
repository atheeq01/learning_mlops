import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_df = train_df.dropna(subset=['content'],axis=0)
test_df = test_df.dropna(subset=['content'],axis=0)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

X_train = train_df['content']
y_train = train_df['sentiment']
X_test = test_df['content']
y_test = test_df['sentiment']


vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)



data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

sparse.save_npz(os.path.join(data_path, "X_train.npz"), X_train_bow)
sparse.save_npz(os.path.join(data_path, "X_test.npz"), X_test_bow)

# Save labels separately
y_train.to_csv(os.path.join(data_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(data_path, "y_test.csv"), index=False)

# -------- save as csv, but it is using such a big data file --------
# train_df = pd.DataFrame(X_train_bow.toarray())
# train_df['sentiment'] = y_train
#
# test_df = pd.DataFrame(X_test_bow.toarray())
# test_df['sentiment'] = y_test
#
# data_path = os.path.join("data", "features")
# os.makedirs(data_path, exist_ok=True)
# train_df.to_csv(os.path.join(data_path, "train.csv"))
# test_df.to_csv(os.path.join(data_path, "test.csv"))

