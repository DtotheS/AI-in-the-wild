from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import csv

os.getcwd()
sn = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_16to21.csv")
pf = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv3_16to21.csv")
pf['content_owner'] = "Politifact"

len(sn) # 10679
len(pf) # 9534
sn.columns
pf.columns

statements = pd.concat([sn.claim,pf.claim],ignore_index=True)
len(statements)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(statements)
arr = X.toarray()

snarr = arr[:10679]
pfarr = arr[10679:]
len(sn) == len(snarr)

''' Evaluation'''
tfidf_sim = cosine_similarity(snarr,pfarr) # sklearn pairwise cosine similarity

sn_same = [any(y >= 0.5 for y in tf) for tf in tfidf_sim]
pf_same = [any(y >= 0.5 for y in tf) for tf in tfidf_sim.T]

sum(sn_same) #663 / 10679 (6.2%)
sum(pf_same) #634 / 9534 (7.0%)

sn['overlap'] = sn_same
pf['overlap'] = pf_same

sim_result = {}
sim_result['website'] = []
sim_result['claim'] = []
sim_result['rating'] = []
sim_result['link'] = []
sim_result['website2'] = []
sim_result['claim2'] = []
sim_result['rating2'] = []
sim_result['link2'] = []

tfidf_sim[0][tfidf_sim[0].argmax()]

# SN based similarity
for i in range(len(sn)):
    if tfidf_sim[i].max() >= 0.5:
        sim_result['website'].append(sn['content_owner'][i])
        sim_result['claim'].append(sn['claim'][i])
        sim_result['rating'].append(sn['rating'][i])
        sim_result['link'].append(sn['url'][i])
        sim_result['website2'].append(pf['content_owner'][tfidf_sim[i].argmax()])
        sim_result['claim2'].append(pf['claim'][tfidf_sim[i].argmax()])
        sim_result['rating2'].append(pf['rating'][tfidf_sim[i].argmax()])
        sim_result['link2'].append(pf['link'][tfidf_sim[i].argmax()])

sum(sn_same) == len(sim_result['link'])

# PF based similarity
for j in range(len(pf)):
    if tfidf_sim.T[j].max() >= 0.5:
        sim_result['website'].append(pf['content_owner'][j])
        sim_result['claim'].append(pf['claim'][j])
        sim_result['rating'].append(pf['rating'][j])
        sim_result['link'].append(pf['link'][j])
        sim_result['website2'].append(sn['content_owner'][tfidf_sim.T[j].argmax()])
        sim_result['claim2'].append(sn['claim'][tfidf_sim.T[j].argmax()])
        sim_result['rating2'].append(sn['rating'][tfidf_sim.T[j].argmax()])
        sim_result['link2'].append(sn['url'][tfidf_sim.T[j].argmax()])

sum(sn_same)+sum(pf_same) == len(sim_result['rating'])

simdf = pd.DataFrame(sim_result)

set(sn['rating'])
set(pf['rating'])
'''
'true' = 'TRUE'
'mostly-true' =  'Mostly True'
'half-true' = 'Mixture'
'barely-true' = 'Mostly False'
'false' = 'FALSE'
'pants-fire' = 'FALSE'
'''

simdf['rating_re'] = simdf['rating']
simdf['rating2_re'] = simdf['rating2']

# simdf['rating_re'].replace(to_replace=['true','mostly-true','half-true','barely-true','false','pants-fire'],value=['TRUE','Mostly True','Mixture','Mostly False','FALSE','FALSE'])
simdf.loc[simdf["rating_re"] == "true", "rating_re"] = 'True'
simdf.loc[simdf["rating_re"] == 'mostly-true', "rating_re"] = 'Mostly True'
simdf.loc[simdf["rating_re"] == 'half-true', "rating_re"] = 'Mixture'
simdf.loc[simdf["rating_re"] == 'barely-true', "rating_re"] = 'Mostly False'
simdf.loc[simdf["rating_re"] == 'false', "rating_re"] = 'FALSE'
simdf.loc[simdf["rating_re"] == 'pants-fire', "rating_re"] = 'FALSE'
set(simdf['rating_re'])

simdf.loc[simdf["rating2_re"] == "true", "rating2_re"] = 'True'
simdf.loc[simdf["rating2_re"] == 'mostly-true', "rating2_re"] = 'Mostly True'
simdf.loc[simdf["rating2_re"] == 'half-true', "rating2_re"] = 'Mixture'
simdf.loc[simdf["rating2_re"] == 'barely-true', "rating2_re"] = 'Mostly False'
simdf.loc[simdf["rating2_re"] == 'false', "rating2_re"] = 'FALSE'
simdf.loc[simdf["rating2_re"] == 'pants-fire', "rating2_re"] = 'FALSE'

simdf["veracity"] = simdf["rating_re"]
simdf["veracity2"] = simdf["rating2_re"]

simdf.loc[simdf["veracity"] == 'True', "veracity"] = 'real'
simdf.loc[simdf["veracity"] == 'Mostly True', "veracity"] = 'real'
simdf.loc[simdf["veracity"] == 'Mostly False', "veracity"] = 'fake'
simdf.loc[simdf["veracity"] == 'FALSE', "veracity"] = 'fake'

simdf.loc[simdf["veracity2"] == 'True', "veracity2"] = 'real'
simdf.loc[simdf["veracity2"] == 'Mostly True', "veracity2"] = 'real'
simdf.loc[simdf["veracity2"] == 'Mostly False', "veracity2"] = 'fake'
simdf.loc[simdf["veracity2"] == 'FALSE', "veracity2"] = 'fake'

simdf['rating_same'] = 0
simdf['veracity_same'] = 0
for i in range(len(simdf)):
    if simdf['rating_re'][i] == simdf['rating2_re'][i]:
        simdf['rating_same'][i] = 1
    if simdf['veracity'][i] == simdf['veracity2'][i]:
        simdf['veracity_same'][i] = 1

simdf['rating_same'].sum() /len(simdf) #896 (69.1%)
simdf['veracity_same'].sum() /len(simdf) #955 (73.6%)
simdf[simdf['website']=="Snopes"]['rating_same'].sum() # 456/663 (68.8%)
simdf[simdf['website']=="Politifact"]['rating_same'].sum() # 440/634 (69.4%)
simdf[simdf['website']=="Snopes"]['veracity_same'].sum() # 486/663 (68.8%)
simdf[simdf['website']=="Politifact"]['veracity_same'].sum() # 469/634 (69.4%)

simdf['rv_dif'] = 0
for i in range(len(simdf)):
    if simdf['rating_same'][i] != simdf['veracity_same'][i]:
        simdf['rv_dif'][i] = 1

simdf['rv_dif'].sum() #59

simdf.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_pf_sim_result.csv",index=False)