from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from collections import Counter
import csv
import re

os.getcwd()
os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")

sn = pd.read_csv("../data/sn_100222.csv")
aap = pd.read_csv("../data/aap_090122.csv")
pf = pd.read_csv("../data/pfv6_16to22.csv")
lg = pd.read_csv("../data/logically_090622_v2.csv")


sn = pd.read_csv("../data/sn_100222.csv")
aap = pd.read_csv("../data/aap_090122.csv")
pf = pd.read_csv("../data/pfv6_16to22.csv")
lg = pd.read_csv("../data/logically_090622_v2.csv")

# lg = lg[lg['location']=="United States"] # Only for US.

sn = sn[sn['page_type']=="Fact Check"]
sn.isnull().sum()
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

pf['site'] = "pf"

sn = sn[sn['claim'].notnull()]

len(sn) # 5932
len(aap) # 827
len(pf) # 5615
len(lg) # 4338

# df = pd.DataFrame()
# df['claim'] = pd.concat([pf['claim'],sn['claim'],lg['title'],aap['title']],ignore_index=True)
# df['site'] = pd.concat([pf['site'],sn['content_owner'],lg['site'],aap['site']],ignore_index=True)
# len(df) == len(pf) + len(sn) + len(lg) + len(aap)

# enc = [0] * len(pf) + [1]*len(sn) + [2]*len(lg) + [3]*len(aap)
# len(enc) == len(df)
# df['label'] = enc

bdf = pd.DataFrame()
bdf['claim'] = pd.concat([pf['claim'],sn['claim']],ignore_index=True)
bdf['site'] = pd.concat([pf['site'],sn['content_owner']],ignore_index=True)
len(bdf) == len(pf) + len(sn)

enc = [0] * len(pf) + [1]*len(sn)
len(enc) == len(bdf)
bdf['label'] = enc

# NER
import spacy
from spacy import displacy

NER = spacy.load("en_core_web_lg")
# !python -m spacy validate
# !python -m spacy download en_core_web_md
# !python -m spacy download en_core_web_sm

# Text Data Cleaning and Model Building with CountVectorizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def cleanText(message):
    # message = message.translate(str.maketrans('', '', string.punctuation))

    # words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    words = [lemmatizer.lemmatize(word.strip("\"\'““”").lower().strip()) for word in word_tokenize(message) if word.lower() not in stopwords.words("english")]
    words = [re.sub('[^A-Za-z0-9]+','',ele) for ele in words]
    # Remove some verbs
    removeli = ['say', 'show', 'u', 's', '2020', '2021', '2022']
    words = [e for e in words if e not in removeli]
    while ("" in words):
        words.remove("")
    m2 = " ".join(words)
    m2 = m2.strip()
    return m2

df["claim_cl"] = df["claim"].apply(cleanText)
bdf["claim_cl"] = bdf["claim"].apply(cleanText)
bdf.head(n=10)

# y = df['label']
# x = df['claim_cl']
by = bdf['label']
bx = bdf['claim_cl']

## EDA1: Wordcloud for each of pf vs. sn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# pd.Series(pfc_words).value_counts().head(n=20)
# pd.Series(snc_words).value_counts().head(n=20)

pfc = " ".join(i for i in bdf[bdf['label']==0]['claim_cl'])
snc = " ".join(i for i in bdf[bdf['label']==1]['claim_cl'])

from wordcloud import WordCloud
pf_wordcloud = WordCloud(width=600, height=400,background_color='white',collocations=False).generate(pfc)
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(pf_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/img/prediction/pf_cl2.png')
plt.show()

sn_cloud=WordCloud(width=600,height=400,background_color='white',collocations=False).generate(snc)
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(sn_cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/img/prediction/sn_cl2.png')
plt.show()

#top 10 pf words
pfcli = word_tokenize(pfc)
sncli = word_tokenize(snc)

pfc_words=np.array(pfcli)
pd.Series(pfc_words).value_counts().head(n=10)

#top 10 sn words
snc_words=np.array(sncli)
pd.Series(snc_words).value_counts().head(n=10)

## EDA 2: Compare proporion of sn vs pf within dataset
plt.figure(figsize=(12,5))
bdf['label'].value_counts().plot(kind='bar',color='blue', label='PF vs. SN')
bdf['label'].value_counts().index.values
bdf['label'].value_counts()
for i in range(len(bdf['label'].value_counts())):
    plt.text(bdf['label'].value_counts().index.values[i], bdf['label'].value_counts()[i], bdf['label'].value_counts()[i], ha = 'center')
# plt.legend(prop={'size': 25})
plt.xticks([0,1],['pf','sn'],fontsize=20,rotation=360)
# plt.xticks(np.arange(0,len(dcnt),5), dname[::5], rotation=90)
# plt.savefig('hw4prop.png')
plt.show()

## FE: CounterVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
# x = cv.fit_transform(x)
bx = cv.fit_transform(bx)

# print(x.shape, y.shape)
print(bx.shape, by.shape)

## Model building
# train test split
from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2) # for 4 categories
bx_train,bx_test,by_train,by_test=train_test_split(bx,by,random_state=0,test_size=0.2) # binary: pf vs sn
'''
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

dtree_model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
dtree_predictions = dtree_model.predict(x_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
'''

## Model Building

## 1. LR
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=10000)
lr.fit(bx_train,by_train)
pred_1=lr.predict(bx_test)

# Evaluation
accuracy_score(by_test,pred_1)
confusion_matrix(by_test,pred_1)
precision_recall_fscore_support(by_test,pred_1, average='binary')


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(bx_train,by_train)
pred_2=nb.predict(bx_test)

# Evaluation
accuracy_score(by_test,pred_2)
confusion_matrix(by_test,pred_2)
precision_recall_fscore_support(by_test,pred_2, average='binary')

from sklearn.svm import SVC
svm=SVC()
svm.fit(bx_train,by_train)
pred_3=svm.predict(bx_test)

# Evaluation
accuracy_score(by_test,pred_3)
confusion_matrix(by_test,pred_3)
precision_recall_fscore_support(by_test,pred_3, average='binary')

'''
# EDA 1: Text length compare: spam vs ham
len_text=[]
for i in df['claim_cl']:
    len_text.append(len(i))

# add text_length as another column
df['text_length']=len_text

import matplotlib.pyplot as plt
#histogram for pf
plt.figure(figsize=(12,5))
df[df['label']==0]['text_length'].plot(bins=35,kind='hist',color='blue',label='spam',alpha=0.5)
plt.legend()
plt.xlabel('message length')
plt.show()

df[df['label']==0]['text_length'].mean()
df[df['label']==1]['text_length'].mean()

# histogram for sn
plt.figure(figsize=(12,5))
df[df['label']==1]['text_length'].plot(bins=35,kind='hist',color='blue',label='spam',alpha=0.5)
plt.legend()
plt.xlabel('message length')
plt.show()

import seaborn as sns
sns.set_style('darkgrid')

f, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.distplot(df[df["spam"] == 1]["text_length"], bins = 20, ax = ax[0])
ax[0].set_xlabel("Spam Message Length", fontsize=25)

sns.distplot(df[df["spam"] == 0]["text_length"], bins = 20, ax = ax[1])
ax[1].set_xlabel("Ham Message Length", fontsize=25)

plt.savefig('hw4len.png')
plt.show()
'''






