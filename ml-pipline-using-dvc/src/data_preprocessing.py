import string
import re
import pandas as pd
from nltk.corpus import stopwords
import spacy
import os

nlp = spacy.load("en_core_web_sm")

url_pattern = re.compile(r'http?://\S+|www\.\S+')
punct_words = re.compile(r'[^\w\s]')
stop_words = set(stopwords.words('english'))
negative_words = {
    'no', 'not', 'nor', 'don', "don't", 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}
custom_stopwords = stop_words - negative_words


def normalize_text(df, text_column="content"):
    print("Starting Batch Normalization ......")
    clean_series = df[text_column].astype(str).str.lower()
    clean_series = clean_series.str.replace(url_pattern, "", regex=True)
    clean_series = clean_series.str.replace(punct_words , "", regex=True)

    clean_texts = []

    for doc in nlp.pipe(clean_series.tolist(), batch_size=1000, n_process=-1):
        token = [
            token.lemma_ for token in doc
            if token.text not in custom_stopwords
               and not token.text.isdigit()
               and token.text.strip() != ''
        ]
        clean_texts.append(" ".join(token))

    df[text_column] = clean_texts
    print("Finished Batch Normalization ......")
    return df



test_data = pd.read_csv("data/raw/test.csv")
train_data = pd.read_csv("data/raw/train.csv")

train_data = normalize_text(train_data)
test_data = normalize_text(test_data)

data_path = os.path.join("data","processed")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)