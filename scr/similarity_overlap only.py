from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

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

'''Word Embeddings'''
# sklearn one-hot-encoding = bag-of-words = sklearn CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
ohe = CountVectorizer(lowercase=True, stop_words='english')
X1 = ohe.fit_transform(statements)
arr1 = X1.toarray()
ohe.get_feature_names_out() # words

# TF-IDF
## TF-IDF vectors are an extension of the one-hot encoding model. Instead of considering the frequency of words in one document, the frequency of words across the whole corpus is taken into account. The big idea is that words that occur a lot everywhere carry very little meaning or significance. For instance, trivial words like “and”, “or”, “is” don’t carry as much significance as nouns and proper nouns that occur less frequently.
## Mathematically, Term Frequency (TF) is the number of times a word appears in a document divided by the total number of words in the document. And Inverse Document Frequency (IDF) = log(N/n)
## Although TF-IDF vectors offer a slight improvement over simple count vectorizing, they still have very high dimensionality and don’t capture semantic relationships.

from sklearn.feature_extraction.text import TfidfVectorizer
ti = TfidfVectorizer()
X2 = ti.fit_transform(statements)
arr2 = X2.toarray()

# SentenceBert
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
arr3 = model.encode(statements)
arr3.shape

''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

pf_true = [bool(ele) for ele in pf['overlap'].values]
fc_true = [bool(ele) for ele in fc['overlap'].values]
xs = np.arange(0,1,0.05)

def evaluation(arr,xs):
    # len(pf)  # 1178: num of PFs
    pfarr = arr[:1178]
    fcarr = arr[1178:]
    sim = cosine_similarity(pfarr,fcarr) # sklearn pairwise cosine similarity
    pf_f1 = []
    fc_f1 = []
    for x in xs:
        pf_y = [any(y > x for y in tf) for tf in sim]
        fc_y = [any(y > x for y in tf) for tf in sim.T]
        pf_e = precision_recall_fscore_support(pf_true, pf_y, average='binary')
        fc_e = precision_recall_fscore_support(fc_true, fc_y, average='binary')
        pf_f1.append(pf_e[2])  # compare f1 score
        fc_f1.append(fc_e[2])  # compare f1 score
    return sim, pf_f1, fc_f1

def img_f1dist(li1,li2,li3):
    import matplotlib.pyplot as plt
    plt.plot(xs, li1, label="Count Vectorizer")
    plt.plot(xs, li2, label="TF-IDF Vectorizer")
    plt.plot(xs, li3, label="Sentence BERT")
    # plt.plot(xs, fc_f1, label="Washington Post")
    plt.ylim([0.1,0.8])
    plt.xlabel("Similarity score")
    plt.ylabel("F1 score for the positive class")
    plt.xticks(np.arange(0,1.1,0.1))
    plt.legend()
    plt.grid(which='major', axis='both')
    # plt.title(title_img)
    plt.show()

ohe_sim, pf_f11, fc_f1 = evaluation(arr1,xs)
tfidf_sim, pf_f12, fc_f1 = evaluation(arr2,xs)
bert_sim, pf_f13, fc_f1 = evaluation(arr3,xs)

img_f1dist(pf_f11,pf_f12,pf_f13)

# Find the max f1-score for each case
# img_f1dist(pf_f1,fc_f1,"TF-IDF Vectorizer")
round(max(pf_f13),4) # 67.11, 71.62, 66.12
round(max(fc_f13),4) # 69.46, 74.58, 63.29
np.argmax(pf_f13)
np.argmax(fc_f13)
xs[np.argmax(pf_f13)] # 0.55, 0.5, 0.9
xs[np.argmax(fc_f13)] # 0.5, 0.4, 0.85

## Confusion Matrix for Politifact max point
x = 0.50 # similarity criterion
pf_y = [any(y>x for y in tf) for tf in tfidf_sim]
# x = 0.90 # similarity criterion
# pf_y = [any(y>x for y in tf) for tf in bert_sim]

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

### Visualization using scikitplot
skplt.metrics.plot_confusion_matrix(fc_true, fc_y)
plt.show()

print(classification_report(fc_true, fc_y))
accuracy = accuracy_score(fc_true, fc_y)
score = precision_recall_fscore_support(fc_true, fc_y, average='weighted')
precision, recall, fscore, k = score
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F score:",fscore)

# Cosine Similarity computes the similarity of two vectors as the cosine of the angle between two vectors. It determines whether two vectors are pointing in roughly the same direction. So if the angle between the vectors is 0 degrees, then the cosine similarity is 1.
# Word embeddings will only give values between 0degree to 90degree (since use only the first quadrant). Thus, the values should be between 0 ~ 1. 1 is cos 0 = similar. so larger is better.
## So cosine similarity is generally preferred over Euclidean distance when working with text data.  The only length-sensitive text similarity use case that comes to mind is plagiarism detection.

