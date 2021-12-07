import numpy as np
import pandas as pd
import os
# use spacy for keyword extraction
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/aidata.csv")
df.columns
df.head
# len(df['text'])
# type(df['text'])

# Use title and text(claim or content)
df['total'] = np.nan

# for the claim, since it is already summarzed, I only use claim.
# for the sources, I used both title and contents.
for i in range(len(df['total'])):
    if df['sourceid'][i] == 1:
        df['total'][i] = df['text'][i]
    else:
        df['total'][i] = df['title'][i] + " " + df['text'][i]

## text cleaning.

nlp = spacy.load("en_core_web_sm")
# stops = stopwords.words("english") # nltk
stops = STOP_WORDS #spacy
def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma.isalpha():
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)

df['spacy_key'] = np.nan
for i in range(len(df['total'])):
    if df['sourceid'][i] == 1:
        df['spacy_key'][i] = str(nlp(df['total'][i]))
    else:
        df['spacy_key'][i] = str(nlp(df['total'][i]).ents) # Named entity Recognition (key word extraction)

# better to clean after key extraction
# for the clain, key exraction make too small set, so it is better not to do.
df['spacy_clean'] = df['spacy_key'].apply(normalize, lowercase=True, remove_stopwords=True)

'''
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
ps = PorterStemmer()
df['spacy_cl'] = np.nan
for i in range(0,len(df['spacy_key'])):
    review = list(df['spacy_key'][i]) # make a list from tuple
    # review = [x for x in df['spacy_key'][i] if str(x).isalpha()] # contains only alphabet
    review = [str(x).lower() for x in review] # lower
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # remove stopwords and stemming
    review = list(set(review)) # remove duplicates
    df['spacy_cl'][i] = review
'''

# Now, we can start to find associative inference pattern.
df.columns

# 1. remove sources after the date of fake news
from datetime import datetime as dt

df['datetime'] = np.nan
for i in range(len(df['date'])):
    df['datetime'][i] = dt.strptime(df['date'][i],"%m/%d/%y")

# select sources before the fake news date
for i in range(len(df[df['id']==1])):
    if df['datetime'][i] <= df[(df['sourceid'] == 1) & (df['id'] == 1)]['datetime'][0]:
        print(df['datetime'][i])

def spl(row):
    li = row.split()
    return li

df['spacy_clean'] = df['spacy_clean'].apply(spl)

## Case 1: Cocaine Mitch
# 2. key word selection
k1 = df['spacy_clean'][0][1] # cocaine
k2 = df['spacy_clean'][0][11] # mitch

k1_idx = []
k2_idx = []
for i in range(len(df[df['id']==1])):
    if df['datetime'][i] <= df[df['sourceid'] == 1]['datetime'][0]:
        if k1 in df['spacy_clean'][i]:
            k1_idx.append(i)

for i in range(len(df[df['id']==1])):
    if df['datetime'][i] <= df[df['sourceid'] == 1]['datetime'][0]:
        if k2 in df['spacy_clean'][i]:
            k2_idx.append(i)

k1_idx
k2_idx

# 3. association
common_keys = []
for i in k1_idx:
    for j in k2_idx:
        if i != j:
            s1 = set(df['spacy_clean'][i])
            s2 = set(df['spacy_clean'][j])
            inter = s1.intersection(s2)
            inter_list = list(inter)
            common_keys.extend(inter_list)

from collections import Counter
x = Counter(common_keys)
x.most_common()

## Case 2: Trump  & (School) lunch (program): Trump - Obama - School lunch program
w1 = df['spacy_clean'][8][1] # trump
w2 = df['spacy_clean'][8][6] # mitch

w1_idx = []
w2_idx = []
for i in range(len(df[df['id']==1]),len(df[df['id']==1])+len(df[df['id']==2])):
    if df['datetime'][i] <= df[df['sourceid'] == 1]['datetime'][8]: # 8 is index for id=2 & sourceid=1
        if w1 in df['spacy_clean'][i]:
            w1_idx.append(i)

for i in range(len(df[df['id']==1]),len(df[df['id']==1])+len(df[df['id']==2])):
    if df['datetime'][i] <= df[df['sourceid'] == 1]['datetime'][8]:
        if w2 in df['spacy_clean'][i]:
            w2_idx.append(i)

w1_idx
w2_idx

# 3. association
common_keys2 = []
for i in w1_idx:
    for j in w2_idx:
        if i != j:
            s1 = set(df['spacy_clean'][i])
            s2 = set(df['spacy_clean'][j])
            inter = s1.intersection(s2)
            inter_list = list(inter)
            common_keys2.extend(inter_list)

from collections import Counter
x2 = Counter(common_keys2)
x2.most_common()


# Need to think how I efficiently check all the documents' connections

