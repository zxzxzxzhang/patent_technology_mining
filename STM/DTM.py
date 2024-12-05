'''
Pre-release Notice

This repository contains code associated with our ongoing research project titled "An insight into the technological landscape and competitive situation of biodegradable PBAT industry based on multidimensional patent analysis". The code is being made available for **review purposes only** and is subject to the following restrictions:

1. Non-commercial use only: This code may only be used for academic or non-commercial purposes.
2. No redistribution or modification**: Redistribution or modification of this code is not permitted until the associated research paper has been officially published.
3. Temporary access: The code in this repository is subject to updates and may change without notice until the final release.

After the publication of the corresponding research paper, we plan to release the code under a more permissive open-source license (e.g., MIT License).

For any questions or specific permissions, please contact zhangx2293@gmail.com with the subject "Pre-release Code Inquiry".

Written by Xiang Zhang
'''


import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# Load data and remove empty abstracts
df = pd.read_excel(r'data.xlsx')
df = df.dropna(subset=['摘要(译)(English)'])
df = df[df['摘要(译)(English)'] != '-']

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Text cleaning function
def clean_step1(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    pattern = re.compile(r'[^a-zA-Z\s]')
    text = pattern.sub(' ', text)  # Remove all symbols and numbers
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [token.lower() for token in tokens]  # Convert to lowercase
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    doc = nlp(' '.join(tokens))  # Process with SpaCy
    cleaned_text = ' '.join([token.lemma_.lower().strip() for token in doc if not token.is_stop])
    return cleaned_text

df['摘要(译)(English)'] = df['摘要(译)(English)'].astype(str).apply(clean_step1)

# Tokenized cleaning function
def clean_step2(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    pattern = re.compile(r'[^a-zA-Z\s]')
    text = pattern.sub(' ', text)  # Remove all symbols and numbers
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [token.lower() for token in tokens]  # Convert to lowercase
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    doc = nlp(text)
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop]
    return tokens

df['摘要(译)(English)'] = df['摘要(译)(English)'].astype(str).apply(clean_step2)

# Convert text to Document-Term Matrix
corpus = df['摘要(译)(English)'].apply(lambda tokens: ' '.join(tokens))
vectorizer = CountVectorizer()
DTM = vectorizer.fit_transform(corpus)

# Convert DTM to DataFrame and save to CSV
DTM_df = pd.DataFrame(DTM.toarray(), columns=vectorizer.get_feature_names_out())
DTM_df.to_csv(r'DTM_new.csv')
