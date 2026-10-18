import re
import pandas as pd 
import csv
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

def clean_headline(row):
    tokens = word_tokenize(row.lower())
    clean_tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in clean_tokens
        if word not in stop_words
    ]
    return " ".join(filtered_tokens)    


clickbait = []

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_headline'] = df["headline"].apply(clean_headline)
    df = df.drop(columns=['headline'])
    return df

y = load_data('clickbait_data.csv')
print(y)