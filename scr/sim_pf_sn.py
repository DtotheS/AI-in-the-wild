from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

os.getcwd()
os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")

pf = pd.read_csv("../data/pfv3_16to21.csv")
sn = pd.read_csv("../data/fcs_16to21.csv") # need to be fixed to sn_16to21

len(pf) # 9534
len(sn) # 11073
statements = pd.concat([pf['claim'],sn['claim']],ignore_index=True)

# sklearn one-hot-encoding = bag-of-words = sklearn CountVectorizer()
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X = vectorizer.fit_transform(statements)
arr = X.toarray()
vectorizer.get_feature_names_out() # words

# TF-IDF
## TF-IDF vectors are an extension of the one-hot encoding model. Instead of considering the frequency of words in one document, the frequency of words across the whole corpus is taken into account. The big idea is that words that occur a lot everywhere carry very little meaning or significance. For instance, trivial words like “and”, “or”, “is” don’t carry as much significance as nouns and proper nouns that occur less frequently.
## Mathematically, Term Frequency (TF) is the number of times a word appears in a document divided by the total number of words in the document. And Inverse Document Frequency (IDF) = log(N/n)
## Although TF-IDF vectors offer a slight improvement over simple count vectorizing, they still have very high dimensionality and don’t capture semantic relationships.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(statements)
arr = X.toarray()
arr_pf = arr[:len(pf)]
arr_sn = arr[len(pf):]

# create_heatmap(cosine_similarity(pfarr,fcarr),pflist,fclist)
# plt.show()

''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
tfidf_sim = cosine_similarity(arr_pf,arr_sn) # sklearn pairwise cosine similarity

x = 0.5
pf_y = [any(y>x for y in tf) for tf in tfidf_sim]
sn_y = [any(y>x for y in tf) for tf in tfidf_sim.T]

sum(pf_y)/len(pf) # 25.0% similar / 10.7% overlap
sum(sn_y)/len(sn) # 29.2% similar / 10.9% overlap