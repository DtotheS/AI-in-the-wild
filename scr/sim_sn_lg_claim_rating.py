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

sn = pd.read_csv("../data/sn_100222.csv")
pf = pd.read_csv("../data/logically_090622_v2.csv")

sn = sn[sn['page_type'] == "Fact Check"]
sn.isnull().sum()

sn['date_published'] = pd.to_datetime(sn['date_published'])
pf['fc_date'] = pd.to_datetime(pf['fc_date'])

max(sn['date_published'].min(), pf['fc_date'].min())

sn = sn[sn['date_published'].between(dt(2019, 5, 17), dt(2022, 8, 31))]
pf = pf[pf['fc_date'].between(dt(2019, 5, 17), dt(2022, 8, 31))]

pf = pf[pf['location'] == 'United States'] # only use: 993
pf = pf.reset_index(drop=True)

sn.isnull().sum()
sn = sn[sn['claim'].notnull()]
sn = sn.reset_index(drop=True)



len(sn)  # 5932
len(pf)  # logically: 993
statements = pd.concat([pf['title'], sn['claim']], ignore_index=True)
len(statements) == len(pf) + len(sn)
statements.isnull().sum()

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

len(arr_pf) == len(pf)
len(arr_sn) == len(sn)
# create_heatmap(cosine_similarity(pfarr,fcarr),pflist,fclist)
# plt.show()

''' Evaluation'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def sim(arr1, arr2, cri):
    tfidf_sim = cosine_similarity(arr1, arr2)  # sklearn pairwise cosine similarity
    x = cri
    y1 = [any(y >= x for y in tf) for tf in tfidf_sim]
    y2 = [any(y >= x for y in tf) for tf in tfidf_sim.T]
    return tfidf_sim, y1, y2, sum(y1), sum(y2), (sum(y1) / len(arr1)) * 100, (
                sum(y2) / len(arr2)) * 100  # gives percentage(%) for arr1 base, arr2 base.


# sn vs. pf
# sim(arr_sn,arr_pf,0.4)
tfidf_sim, y1, y2, sumy1, sumy2, y1p, y2p = sim(arr_sn, arr_pf, 0.5)

pf['content_owner'] = "Logically"

sn['overlap'] = y1
pf['overlap'] = y2

sn_sim_result = {}
sn_sim_result['sim_score'] = []
sn_sim_result['website'] = []
sn_sim_result['claim'] = []
sn_sim_result['rating'] = []
sn_sim_result['link'] = []
sn_sim_result['date'] = []
sn_sim_result['website2'] = []
sn_sim_result['claim2'] = []
sn_sim_result['rating2'] = []
sn_sim_result['link2'] = []
sn_sim_result['date2'] = []

tfidf_sim[0][tfidf_sim[0].argmax()]
# SN based similarity
for i in range(len(sn)):
    if tfidf_sim[i].max() >= 0.5:
        print(tfidf_sim[i].max())
        sn_sim_result['sim_score'].append(round(tfidf_sim[i].max(), 3))
        sn_sim_result['website'].append(sn['content_owner'][i])
        sn_sim_result['claim'].append(sn['claim'][i])
        sn_sim_result['rating'].append(sn['rating'][i])
        sn_sim_result['link'].append(sn['link'][i])
        sn_sim_result['date'].append(sn['date_published'][i])
        sn_sim_result['website2'].append(pf['content_owner'][tfidf_sim[i].argmax()])
        sn_sim_result['claim2'].append(pf['title'][tfidf_sim[i].argmax()])
        sn_sim_result['rating2'].append(pf['rating'][tfidf_sim[i].argmax()])
        sn_sim_result['link2'].append(pf['link'][tfidf_sim[i].argmax()])
        sn_sim_result['date2'].append(pf['fc_date'][tfidf_sim[i].argmax()])

sum(y1) == len(sn_sim_result['link'])
sn_simdf = pd.DataFrame(sn_sim_result)

pf_sim_result = {}
pf_sim_result['sim_score'] = []
pf_sim_result['website'] = []
pf_sim_result['claim'] = []
pf_sim_result['rating'] = []
pf_sim_result['link'] = []
pf_sim_result['date'] = []
pf_sim_result['website2'] = []
pf_sim_result['claim2'] = []
pf_sim_result['rating2'] = []
pf_sim_result['link2'] = []
pf_sim_result['date2'] = []

# PF based similarity
for j in range(len(pf)):
    if tfidf_sim.T[j].max() >= 0.5:
        pf_sim_result['sim_score'].append(round(tfidf_sim.T[j].max(), 3))
        pf_sim_result['website'].append(pf['content_owner'][j])
        pf_sim_result['claim'].append(pf['title'][j])
        pf_sim_result['rating'].append(pf['rating'][j])
        pf_sim_result['link'].append(pf['link'][j])
        pf_sim_result['date'].append(pf['fc_date'][j])
        pf_sim_result['website2'].append(sn['content_owner'][tfidf_sim.T[j].argmax()])
        pf_sim_result['claim2'].append(sn['claim'][tfidf_sim.T[j].argmax()])
        pf_sim_result['rating2'].append(sn['rating'][tfidf_sim.T[j].argmax()])
        pf_sim_result['link2'].append(sn['link'][tfidf_sim.T[j].argmax()])
        pf_sim_result['date2'].append(sn['date_published'][tfidf_sim.T[j].argmax()])

sum(y2) == len(pf_sim_result['rating'])
pf_simdf = pd.DataFrame(pf_sim_result)

set(sn['rating'])
set(pf['rating'])
'''
'TRUE' = 'True'
'PARTLY TRUE' =  'Mostly True'
'NONE' = 'No Rating'
'MISLEADING' = 'Mostly False'
'FALSE' = 'False'
'UNVERIFIABLE' =  'Unproven'
'''

sn_simdf['unified_rating'] = sn_simdf['rating']
sn_simdf['unified_rating2'] = sn_simdf['rating2']
pf_simdf['unified_rating'] = pf_simdf['rating']
pf_simdf['unified_rating2'] = pf_simdf['rating2']

# simdf['rating_re'].replace(to_replace=['true','mostly-true','half-true','barely-true','false','pants-fire'],value=['TRUE','Mostly True','Mixture','Mostly False','FALSE','FALSE'])
import itertools

df_li = [sn_simdf, pf_simdf]

for dfname in df_li:
    dfname.loc[dfname["unified_rating"] == 'TRUE', "unified_rating"] = 'True'
    dfname.loc[dfname["unified_rating"] == 'PARTLY TRUE', "unified_rating"] = 'Mostly True'
    dfname.loc[dfname["unified_rating"] == 'UNVERIFIABLE', "unified_rating"] = 'Unproven'
    dfname.loc[dfname["unified_rating"] == 'MISLEADING', "unified_rating"] = 'Mostly False'
    dfname.loc[dfname["unified_rating"] == 'FALSE', "unified_rating"] = 'False'
    dfname.loc[dfname["unified_rating"] == 'NONE', "unified_rating"] = 'No Rating'

    dfname.loc[dfname["unified_rating2"] == 'TRUE', "unified_rating2"] = 'True'
    dfname.loc[dfname["unified_rating2"] == 'PARTLY TRUE', "unified_rating2"] = 'Mostly True'
    dfname.loc[dfname["unified_rating2"] == 'UNVERIFIABLE', "unified_rating2"] = 'Unproven'
    dfname.loc[dfname["unified_rating2"] == 'MISLEADING', "unified_rating2"] = 'Mostly False'
    dfname.loc[dfname["unified_rating2"] == 'FALSE', "unified_rating2"] = 'False'
    dfname.loc[dfname["unified_rating2"] == 'NONE', "unified_rating2"] = 'No Rating'

set(sn_simdf['unified_rating2'])
set(pf_simdf['unified_rating'])

sn_simdf["veracity"] = sn_simdf["unified_rating"]
sn_simdf["veracity2"] = sn_simdf["unified_rating2"]
pf_simdf["veracity"] = pf_simdf["unified_rating"]
pf_simdf["veracity2"] = pf_simdf["unified_rating2"]

for dfname in df_li:
    dfname.loc[dfname["veracity"] == 'True', "veracity"] = 'real'
    dfname.loc[dfname["veracity"] == 'Mostly True', "veracity"] = 'real'
    dfname.loc[dfname["veracity"] == 'Mostly False', "veracity"] = 'fake'
    dfname.loc[dfname["veracity"] == 'False', "veracity"] = 'fake'

    dfname.loc[dfname["veracity2"] == 'True', "veracity2"] = 'real'
    dfname.loc[dfname["veracity2"] == 'Mostly True', "veracity2"] = 'real'
    dfname.loc[dfname["veracity2"] == 'Mostly False', "veracity2"] = 'fake'
    dfname.loc[dfname["veracity2"] == 'False', "veracity2"] = 'fake'

for dfname in df_li:
    dfname['rating_same'] = 0
    dfname['veracity_same'] = 0
    for i in range(len(dfname)):
        if dfname['unified_rating'][i] == dfname['unified_rating2'][i]:
            dfname['rating_same'][i] = 1
        if dfname['veracity'][i] == dfname['veracity2'][i]:
            dfname['veracity_same'][i] = 1

sn_simdf.name = "Snopes"
pf_simdf.name = "Logically"

for dfname in df_li:
    print(dfname.name + ": # rating same, # veracity same, % rating same, % veracity same ")
    print(dfname['rating_same'].sum(),
          dfname['veracity_same'].sum(),
          (dfname['rating_same'].sum() / len(dfname)) * 100,
          (dfname['veracity_same'].sum() / len(dfname)) * 100)

for dfname in df_li:
    dfname['rat_ver_dif'] = 0
    for i in range(len(dfname)):
        if dfname['rating_same'][i] != dfname['veracity_same'][i]:
            dfname['rat_ver_dif'][i] = 1

len(sn_simdf)
len(pf_simdf)
pf_simdf['rat_ver_dif'].sum()  # 9

##### Combine two Dataframes
all_simdf = sn_simdf.append(pf_simdf)
len(all_simdf)
all_simdf.isnull().sum()

all_simdf.groupby('website').sum()
all_simdf.groupby('website').count()

likert = ['True', 'Mostly True', 'Mixture', 'Mostly False', 'False']
sn[sn['rating'].isin(likert)]  # only calculate FCs among "likert"
# all_simdf[(all_simdf['veracity'] == "fake") | (all_simdf['veracity'] == "real") | (all_simdf['veracity'] == "Mixture")].groupby('website').count()
# all_simdf[(all_simdf['veracity'] == "fake") | (all_simdf['veracity'] == "real") | (all_simdf['veracity'] == "Mixture")].groupby('website').sum()

all_simdf[all_simdf['unified_rating'].isin(likert)]
all_simdf[all_simdf['unified_rating'].isin(likert)].groupby('website').sum()
all_simdf[all_simdf['unified_rating'].isin(likert)].groupby('website').count()
all_simdf.groupby('website').sum()
all_simdf.groupby('website').count()

sn_simdf['rating_same'].sum()

all_simdf = all_simdf.reset_index()
all_simdf['days_delta'] = None
for i in range(len(all_simdf)):
    all_simdf['days_delta'][i] = int((all_simdf['date'][i] - all_simdf['date2'][i]).days)

all_simdf['date'][0] - all_simdf['date'][1]

all_simdf.groupby('veracity_same').mean()
all_simdf[(all_simdf['rating_same'] == 1) & (all_simdf['website'] == "Snopes")]['days_delta'].mean() # positive means debunked later.
all_simdf[(all_simdf['rating_same'] == 0) & (all_simdf['website'] == "Snopes")]['days_delta'].mean()

all_simdf.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_lg_sim_result.csv", index=False)

snpf = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_pf_sim_result_v3.csv")
snlg = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_lg_sim_result.csv")

snpf[snpf['website'] == "Snopes"].groupby('rating_same').mean()
snlg[snpf['website'] == "Snopes"].groupby('rating_same').mean()

snpf[snpf['website'] == "Snopes"].groupby('veracity_same').mean()
snlg[snpf['website'] == "Snopes"].groupby('veracity_same').mean()