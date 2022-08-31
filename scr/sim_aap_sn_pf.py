from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

os.getcwd()
os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")

aap = pd.read_csv("../data/aap_083022_19_21.csv")
pf = pd.read_csv("../data/pfv3_16to21.csv")
sn = pd.read_csv("../data/sn_16to21.csv")

aap.isnull().sum()

pf = pf[pf['fc_year'].between(2019,2021)]
pf.columns
sn.columns
sn = sn[sn['yearp'].between(2019,2021)]
len(aap) # 612
len(pf) # 5161
len(sn) # 5452

statements = pd.concat([aap['title'],pf['claim'],sn['claim']],ignore_index=True)
len(statements) == len(aap) + len(pf) + len(sn)
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
arr_sn = arr[len(aap)+len(pf):]

len(arr_aap)==len(aap)
len(arr_pf)==len(pf)
len(arr_sn)==len(sn)
# create_heatmap(cosine_similarity(pfarr,fcarr),pflist,fclist)
# plt.show()

''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
tfidf_sim = cosine_similarity(arr_aap,arr_pf) # sklearn pairwise cosine similarity

x = 0.5
aap_y = [any(y>x for y in tf) for tf in tfidf_sim]
pf_y = [any(y>x for y in tf) for tf in tfidf_sim.T]

(sum(aap_y)/len(aap))*100 # 1.63% overlap (x=0.5) / 3.92% similar (x=0.4)
(sum(pf_y)/len(pf))*100 # 0.19% overlap (x=0.5) / 0.48% similar (x=0.4)

tfidf_sim2 = cosine_similarity(arr_aap,arr_sn) # aap vs. sn

x = 0.5
aap_y2 = [any(y>x for y in tf) for tf in tfidf_sim2]
sn_y2 = [any(y>x for y in tf) for tf in tfidf_sim2.T]

(sum(aap_y2)/len(aap))*100 # 0.49% overlap (x=0.5) / 1.80% similar (x=0.4)
(sum(sn_y2)/len(sn))*100 # 0.06% overlap (x=0.5) / 0.33% similar (x=0.4)

tfidf_sim3 = cosine_similarity(arr_pf,arr_sn) # pf vs. sn
x = 0.5
pf_y3 = [any(y>x for y in tf) for tf in tfidf_sim3]
sn_y3 = [any(y>x for y in tf) for tf in tfidf_sim3.T]

(sum(pf_y3)/len(pf))*100 # 6.45% overlap (x=0.5) / 12.59% similar (x=0.4)
(sum(sn_y3)/len(sn))*100 # 6.12% overlap (x=0.5) / 12.00% similar (x=0.4)
