# Reference: https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python

# !pip install -U pip setuptools wheel
# !pip install -U 'spacy[apple]'
# !python -m spacy download en_core_web_trf

from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

os.getcwd()
os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")

zipf = "/Users/agathos/DtotheS/AI-in-the-wild/data/lim2018_files.zip"
with ZipFile(zipf,'r') as zip:
    zip.printdir() # printing all the contents of the zip file
    # zip.extractall("/Users/agathos/DtotheS/AI-in-the-wild/data/lim2018") # extract to a folder
    listf = zip.namelist()
    finaldtm = pd.read_csv(zip.open(listf[0]))
    fc = pd.read_csv(zip.open(listf[1]))
    murky = pd.read_csv(zip.open(listf[2]))
    overlap = pd.read_csv(zip.open(listf[3]))
    pf = pd.read_csv(zip.open(listf[4]))
    overlapdtm = pd.read_csv(zip.open(listf[5]))
del zip

statements = pd.concat([pf.statement,fc.statement],ignore_index=True)
list(set(fc.category))


fc['overlap'] = None
fc['overmurky'] = None
for i in range(len(fc)):
    if fc['category'][i] == 'overlap':
        fc['overlap'][i] = 1
        fc['overmurky'][i] = 1
    elif fc['category'][i] == 'murky':
        fc['overlap'][i] = 0
        fc['overmurky'][i] = 1
    else:
        fc['overlap'][i] = 0
        fc['overmurky'][i] = 0

pf['overlap'] = None
pf['overmurky'] = None
for i in range(len(pf)):
    if pf['category'][i] == 'overlap':
        pf['overlap'][i] = 1
        pf['overmurky'][i] = 1
    elif pf['category'][i] == 'murky':
        pf['overlap'][i] = 0
        pf['overmurky'][i] = 1
    else:
        pf['overlap'][i] = 0
        pf['overmurky'][i] = 0

'''
len(murky) # 77
len(overlap) # 56
fc['overlap'].sum() # 77 correct
fc['overmurky'].sum() # 77 + 56 = 133 != 130..... not correct..
pf['overlap'].sum() #correct
pf['overmurky'].sum() # correct

fc[fc['category']=="murky"] # 53 mismatch with murky file
## there are some duplicate murkies in the murky file for FCs: e.g., row (1)39 & 40, (2) 48&49, (3) 50&51=> Each pair is the same FCs.
murky.columns
murky.PFstatement[39] # Dealing with Governor's tax cut
murky.PFstatement[40] # Dealing with Governor's hole + surplus
murky.FCstatement[39] == murky.FCstatement[40] # True: Dealing with Governor's tax cut + hole + surplus
murky.FCurl[39] == murky.FCurl[40]
murky.FCstatement[48] == murky.FCstatement[49]
murky.FCurl[48] == murky.FCurl[49]
murky.FCstatement[50] == murky.FCstatement[51]
murky.FCurl[50] == murky.FCurl[51]

len(set(murky.FCurl)) # This should be 53, but it is 52. I found that there are 1 url error.
from collections import Counter
len([k for k,v in Counter(list(murky.FCstatement.values)).items() if v >1]) # indeed, there are 3 statements when we check the statement
len([k for k,v in Counter(list(murky.FCurl.values)).items() if v >1]) # However, there are 4 duplicates urls

[k for k,v in Counter(list(murky.FCstatement.values)).items() if v >1]

[k for k,v in Counter(list(murky.FCurl.values)).items() if v >1]
murky.FCurl[39]
murky.FCurl[48]
murky.FCurl[50]

murky[murky.FCurl == [k for k,v in Counter(list(murky.FCurl.values)).items() if v >1][0]] # These two have the same urls, but different statements.
murky.iloc[20] 
murky.iloc[25]
# Links are not working, so I could not figure out which one is correct.
'''


'''
import seaborn as sns
import matplotlib.pyplot as plt

def creat_heatmap(similarity,labels_ind,labels_col, cmap="YlGnBu"):
    df = pd.DataFrame(similarity)
    df.columns = labels_col
    df.index = labels_ind
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(df, cmap=cmap)
'''

'''Word Embeddings'''

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

len(pf) #1178
pfarr = arr[:1178]
fcarr = arr[1178:]

'''
creat_heatmap(cosine_similarity(pfarr,fcarr),pf.id,fc.id)
plt.show()
'''

''' Word2Vec: Works very poor
import spacy
# !python -m spacy download en_core_web_md
# !python -m spacy download en_core_web_lg
# !python -m spacy download en_vectors_web_md
# !python -m spacy download en_core_web_trf
nlp = spacy.load('en_core_web_lg')
docs = [nlp(statement) for statement in statements]
pfarr = [sent.vector for sent in docs[:1178]]
fcarr = [sent.vector for sent in docs[1178:]]

similarity = []
for i in range(1178):
    row = []
    for j in range(1178,1503):
        row.append(docs[i].similarity(docs[j]))
    similarity.append(row)
similarity = np.array(similarity)

# label: overlap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

xs = np.arange(0,1,0.05)
pf_true = [bool(ele) for ele in pf['overlap'].values]
fc_true = [bool(ele) for ele in fc['overlap'].values]
pf_f1 = []
fc_f1 = []
for x in xs:
    pf_y = [any(y > x for y in tf) for tf in similarity]
    fc_y = [any(y > x for y in tf) for tf in similarity.T]
    pf_e = precision_recall_fscore_support(pf_true, pf_y, average='binary')
    fc_e = precision_recall_fscore_support(fc_true, fc_y, average='binary')
    pf_f1.append(pf_e[2]) # compare f1 score
    fc_f1.append(fc_e[2]) # compare f1 score

import matplotlib.pyplot as plt
plt.plot(xs,pf_f1,label = "politifact")
plt.plot(xs,fc_f1,label = "Washington Post")
plt.legend()
plt.show()
'''


''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
tfidf_sim = cosine_similarity(pfarr,fcarr) # sklearn pairwise cosine similarity
xs = np.arange(0,1,0.05)

# label: overlap
pf_true = [bool(ele) for ele in pf['overlap'].values]
fc_true = [bool(ele) for ele in fc['overlap'].values]
pf_f1 = []
fc_f1 = []
for x in xs:
    pf_y = [any(y > x for y in tf) for tf in tfidf_sim]
    fc_y = [any(y > x for y in tf) for tf in tfidf_sim.T]
    pf_e = precision_recall_fscore_support(pf_true, pf_y, average='binary')
    fc_e = precision_recall_fscore_support(fc_true, fc_y, average='binary')
    pf_f1.append(pf_e[2]) # compare f1 score
    fc_f1.append(fc_e[2]) # compare f1 score

import matplotlib.pyplot as plt
plt.plot(xs,pf_f1,label = "politifact")
plt.plot(xs,fc_f1,label = "Washington Post")
plt.legend()
plt.show()

max(pf_f1)
max(fc_f1)
np.argmax(pf_f1) # x=0.5
np.argmax(fc_f1) # x=0.4
xs[np.argmax(pf_f1)]
xs[np.argmax(fc_f1)]

## Confusion Matrix for Politifact max point
x = 0.50 # similarity criterion
pf_y = [any(y>x for y in tf) for tf in tfidf_sim]
# pf_y.index(True) # Note: The index() method only returns the first occurrence of the matching element.
# for i in range(len(pf_y)):
#     if pf_y[i] == True:
#         print(i)
# sum(pf_y)
conf_matrix = confusion_matrix(pf_true,pf_y)
precision_recall_fscore_support(pf_true,pf_y, average='binary')

# Print the confusion matrix using Matplotlib: Politifact
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

### Visualization using scikitplot
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_fscore_support,accuracy_score
skplt.metrics.plot_confusion_matrix(pf_true, pf_y)
plt.show()

print(classification_report(pf_true, pf_y))

accuracy = accuracy_score(pf_true, pf_y)
score = precision_recall_fscore_support(pf_true, pf_y, average='weighted')
precision, recall, fscore, k = score
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F score:",fscore)

# Print the confusion matrix using Matplotlib: Factchecker
x = 0.40 # similarity criterion
fc_y = [any(y>x for y in tf) for tf in tfidf_sim.T]
conf_matrix = confusion_matrix(fc_true,fc_y)
precision_recall_fscore_support(fc_true,fc_y, average='binary')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# label: overlap + murky
pf_true_om = [bool(ele) for ele in pf['overmurky'].values]
fc_true_om = [bool(ele) for ele in fc['overmurky'].values]
pf_f1_om = []
fc_f1_om = []
for x in xs:
    pf_y = [any(y > x for y in tf) for tf in tfidf_sim]
    fc_y = [any(y > x for y in tf) for tf in tfidf_sim.T]
    pf_e = precision_recall_fscore_support(pf_true_om, pf_y, average='binary')
    fc_e = precision_recall_fscore_support(fc_true_om, fc_y, average='binary')
    pf_f1_om.append(pf_e[2]) # compare f1 score
    fc_f1_om.append(fc_e[2]) # compare f1 score

import matplotlib.pyplot as plt
plt.plot(xs,pf_f1_om,label = "politifact")
plt.plot(xs,fc_f1_om,label = "Washington Post")
plt.legend()
plt.show()

np.argmax(pf_f1_om) # x=0.4
np.argmax(fc_f1_om) # x=0.35
xs[8]

## Confusion Matrix for Politifact max point
x = 0.40 # similarity criterion
pf_y = [any(y>x for y in tf) for tf in tfidf_sim]
# pf_y.index(True) # Note: The index() method only returns the first occurrence of the matching element.
# for i in range(len(pf_y)):
#     if pf_y[i] == True:
#         print(i)
# sum(pf_y)
conf_matrix = confusion_matrix(pf_true_om,pf_y)
precision_recall_fscore_support(pf_true_om,pf_y, average='binary')

# Print the confusion matrix using Matplotlib: Politifact
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# Print the confusion matrix using Matplotlib: Factchecker
x = 0.35 # similarity criterion
fc_y = [any(y>x for y in tf) for tf in tfidf_sim.T]
conf_matrix = confusion_matrix(fc_true_om,fc_y)
precision_recall_fscore_support(fc_true_om,fc_y, average='binary')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

''' not necessary part. use category column.
pf.columns
overlap.columns
type(overlap.PFurl.values) #ndarray => "a" in s
type(overlap.PFurl) #Series => s.isin(["a","b"])

## add overlap label for pf and df.
pf['overlap'] = None
dd = 0
for i in pf.url:
    if i in overlap.PFurl.values:
        pf['overlap'][dd] = 1
        dd +=1
    else:
        pf['overlap'][dd] = 0
        dd +=1
pf['overlap'].sum()

fc['overlap'] = None
dd = 0
for i in fc.url:
    if i in overlap.FCurl.values:
        fc['overlap'][dd] = 1
        dd +=1
    else:
        fc['overlap'][dd] = 0
        dd +=1
fc['overlap'].sum() # should be 77 but only 71. 6 are missing.
len(fc.url) - len(set(fc.url)) # 16 duplicates in total
fc[fc['overlap']==1]['url'].values # 71
overlap.FCurl.values # 77

for i in overlap.FCurl.values:
    if i not in fc[fc['overlap']==1]['url'].values:
        print(i) # there are 12 urls which are not contained in fc dataset.

len(fc[fc['overlap']==1]['url'].values) - len(set(fc[fc['overlap']==1]['url'].values)) # there are 7 duplicates.

import collections
print([item for item, count in collections.Counter(fc[fc['overlap']==1]['url'].values).items() if count > 1]) # 3 duplicates
collections.Counter(fc[fc['overlap']==1]['url'].values) # there are 4 urls duplicated 3,3,3,2 times. Thus, total 7 were removed from set len check.
print([item for item, count in collections.Counter(pf[pf['overlap']==1]['url'].values).items() if count > 1]) # no duplicate urls

# fc.statement
# overlap.FCstatement
## So, I tried statement for fc data to label overlap. But it turned out even worse.
'''


''' Distance Metrics'''
# Jaccard Index: treats the data objects like sets. the size of the intersection of two sets divided by the size of the union.
## Jaccard similarity is rarely used when working with text data as it does not work with text embeddings. This means that is limited to assessing the lexical similarity of text, i.e., how similar documents are on a word level.


# Euclidean Distance = L2 norm < Minkowski distance
from math import sqrt, pow, exp

def euclidean_dist(x,y):
    return sqrt(sum((a-b)**2 for a, b in zip(x,y)))

euclidean_dist(arr[0],arr[1])
## See, the problem with using distance is that it’s hard to make sense if there is nothing to compare to. The distances can vary from 0 to infinity, we need to use some way to normalize them to the range of 0 to 1.
## Euler's constant:
def distance_to_similarity(distance):
    return 1/exp(distance)

distance_to_similarity(euclidean_dist(arr[0],arr[1])) # between 0 to 1. larger is better (less far)

# cosine similarity
def squared_sum(x):
    #round-up and show 3 decimals
    return round(sqrt(sum([a**2 for a in x])))

def cos_sim(x,y):
    '''
    return cosine similarity between two lists
    cos(x,y) = x*y / (||x||*||y||)
    '''
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = squared_sum(x)*squared_sum(y)
    return round(numerator/float(denominator),3)

cos_sim(arr[0],arr[1])
cos_sim(arr)
# Cosine Similarity computes the similarity of two vectors as the cosine of the angle between two vectors. It determines whether two vectors are pointing in roughly the same direction. So if the angle between the vectors is 0 degrees, then the cosine similarity is 1.
# Word embeddings will only give values between 0degree to 90degree (since use only the first quadrant). Thus, the values should be between 0 ~ 1. 1 is cos 0 = similar. so larger is better.

## So cosine similarity is generally preferred over Euclidean distance when working with text data.  The only length-sensitive text similarity use case that comes to mind is plagiarism detection.

