from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from datetime import datetime as dt

os.getcwd()
os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")

sn = pd.read_csv("../data/sn_091922.csv")
aap = pd.read_csv("../data/aap_090122.csv")
pf = pd.read_csv("../data/politifact_v3_072122.csv")
lg = pd.read_csv("../data/logically_090622_v2.csv")

sn = sn[sn['page_type']=="Fact Check"]
sn = sn[sn['rating'].isnull()==False]
len(sn)
len(pf)
len(aap)
len(lg)
sn['date_published'] = pd.to_datetime(sn['date_published'])
aap['fc_date'] = pd.to_datetime(aap['fc_date'])
pf['fc_date'] = pd.to_datetime(pf['fc_date'])
lg['fc_date'] = pd.to_datetime(lg['fc_date'])

max(sn['date_published'].min(),
aap['fc_date'].min(),
pf['fc_date'].min(),
lg['fc_date'].min())


sn = sn[sn['date_published'].between(dt(2019,5,17),dt(2022,8,31))]
aap = aap[aap['fc_date'].between(dt(2019,5,17),dt(2022,8,31))]
pf = pf[pf['fc_date'].between(dt(2019,5,17),dt(2022,8,31))]
lg = lg[lg['fc_date'].between(dt(2019,5,17),dt(2022,8,31))]

len(sn) # 5933
len(aap) # 827
len(pf) # 5615
len(lg) # 4338

lg.isnull().sum()
aap.isnull().sum()

statements = pd.concat([aap['title'],pf['claim'],sn['claim'],lg['title']],ignore_index=True)
len(statements) == len(aap) + len(pf) + len(sn) + len(lg)
statements.isnull().sum()

# TF-IDF
## TF-IDF vectors are an extension of the one-hot encoding model. Instead of considering the frequency of words in one document, the frequency of words across the whole corpus is taken into account. The big idea is that words that occur a lot everywhere carry very little meaning or significance. For instance, trivial words like “and”, “or”, “is” don’t carry as much significance as nouns and proper nouns that occur less frequently.
## Mathematically, Term Frequency (TF) is the number of times a word appears in a document divided by the total number of words in the document. And Inverse Document Frequency (IDF) = log(N/n)
## Although TF-IDF vectors offer a slight improvement over simple count vectorizing, they still have very high dimensionality and don’t capture semantic relationships.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(statements)
arr = X.toarray()
arr_aap = arr[:len(aap)]
arr_pf = arr[len(aap):len(aap)+len(pf)]
arr_sn = arr[len(aap)+len(pf):len(aap)+len(pf)+len(sn)]
arr_lg = arr[len(aap)+len(pf)+len(sn):]

len(arr_aap)==len(aap)
len(arr_pf)==len(pf)
len(arr_sn)==len(sn)
len(arr_lg)==len(lg)
# create_heatmap(cosine_similarity(pfarr,fcarr),pflist,fclist)
# plt.show()

''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def sim(arr1,arr2,cri):
    tfidf_sim = cosine_similarity(arr1,arr2) # sklearn pairwise cosine similarity
    x = cri
    y1 = [any(y>x for y in tf) for tf in tfidf_sim]
    y2 = [any(y>x for y in tf) for tf in tfidf_sim.T]
    return (sum(y1)/len(arr1))*100, (sum(y2)/len(arr2))*100 #gives percentage(%) for arr1 base, arr2 base.

# sn vs. pf
sim(arr_sn,arr_pf,0.4)
sim(arr_sn,arr_pf,0.5)

# sn vs. aap
sim(arr_sn,arr_aap,0.4)
sim(arr_sn,arr_aap,0.5)

# sn vs. lg
sim(arr_sn,arr_lg,0.4)
sim(arr_sn,arr_lg,0.5)

# pf vs. aap
sim(arr_pf,arr_aap,0.4)
sim(arr_pf,arr_aap,0.5)

# pf vs. lg
sim(arr_pf,arr_lg,0.4)
sim(arr_pf,arr_lg,0.5)

# aap vs. lg
sim(arr_aap,arr_lg,0.4)
sim(arr_aap,arr_lg,0.5)

