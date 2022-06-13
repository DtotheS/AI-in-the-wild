import numpy as np
import pandas as pd
import os
# use spacy for keyword extraction
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from datetime import datetime as dt
import csv

os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/aidata.csv")
# df.columns
# df.head
# len(df['text'])
# type(df['text'])

# Use title and text(claim or content)
df['total'] = np.nan

# for the claim, since it is already summarized, I only use claim.
# for the sources, I used both title and contents.
for i in range(len(df['total'])):
    if df['sourceid'][i] == 1:
        df['total'][i] = df['text'][i]
    else:
        df['total'][i] = df['title'][i] #+ " " + df['text'][i]

## text cleaning. (did after keywrod extraction)

nlp = spacy.load("en_core_web_sm")
# stops = stopwords.words("english") # nltk
stops = STOP_WORDS #spacy
def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower() # lower char
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip() # lemmatization
        if lemma.isalpha(): # only choose alphabet
            if not remove_stopwords or (remove_stopwords and lemma not in stops): #remove stop words
                lemmatized.append(lemma)
    return " ".join(lemmatized)

df['spacy_key'] = np.nan
for i in range(len(df['total'])):
    if df['sourceid'][i] == 1:
        df['spacy_key'][i] = str(nlp(df['total'][i]))
    else:
        df['spacy_key'][i] = str(nlp(df['total'][i])) #.ents) # deleted Named Entity Recognition (key word extraction)

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
# df.columns

# 1. remove sources after the date of fake news


df['datetime'] = np.nan
for i in range(len(df['date'])):
    df['datetime'][i] = dt.strptime(df['date'][i],"%m/%d/%y")

# # select sources before the fake news date
# for i in range(len(df[df['id']==1])):
#     if df['datetime'][i] <= df[(df['sourceid'] == 1) & (df['id'] == 1)]['datetime'][0]:
#         print(df['datetime'][i])

## Removed duplicate keywords for each source.
def spl(row):
    # Split keywords by pace and make a list of words.
    li = row.split()
    # Remove duplicate keywords for each list
    li = list(set(li))
    return li

df['spacy_clean'] = df['spacy_clean'].apply(spl)
# df['spacy_clean'][8]


## Find ai in the wild

# 2. key word selection
# k1 = df['spacy_clean'][0][1] # cocaine
# k2 = df['spacy_clean'][0][11] # mitch

'''
# 3. find ai pattern for all ids with random key words.
import random

common_keys = { }
for i in range(1,max(df['id'])+1):
    # arbitrary choose keyword the first and second one. Need to be modified #
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    rd = random.sample(range(0,len(df[(df['id']==i)&(df['sourceid']==1)]['spacy_clean'].reset_index(drop=True)[0])),2) # randomly generate two numbers based on the maximum number of keywords for the claim of the fake news
    k1 = df[(df['id']==i)&(df['sourceid']==1)]['spacy_clean'].reset_index(drop=True)[0][rd[0]] # df of "id=1 &source=1" will give only one row ([0]) and want to prent the first [0][0] and the second [0][1] keywords for now.
    k2 = df[(df['id']==i)&(df['sourceid']==1)]['spacy_clean'].reset_index(drop=True)[0][rd[1]]
    #############################################################################
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'][foo1]:
                k1_idx.append(foo1)
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['spacy_clean'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['spacy_clean'][k1j])
                set2 = set(df['spacy_clean'][k2j])
                inter = set1.intersection(set2)
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    common_keys["id%s" %i] = (k1,k2,ordered_x)

common_keys
'''
# 3. Finding an ai-pattern function.
## Removed A or C from the candidates of B.

def ai_pattern(id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'][foo1]:
                k1_idx.append(foo1)
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['spacy_clean'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['spacy_clean'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['spacy_clean'][k1j])
                set2 = set(df['spacy_clean'][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)

'''
## Case 1: Cocaine & Ping May & Mitch
c1_1 = df['spacy_clean'][0][1] # cocaine
c1_2 = df['spacy_clean'][0][11] # mitch
ai_pattern(1,c1_1,c1_2)

## Case 2: Trump & Obama & School Lunch Program
c2_1 = df['spacy_clean'][8][1] # trump
c2_2 = df['spacy_clean'][8][5] # school
ai_pattern(2,c2_1,c2_2)

## Case 3: obama & Gordon Ernst & college admissions bribery scandal
c3_idx = df[(df['id']==3)&(df['sourceid']==1)]['spacy_clean'].index[0]
c3_1 = df['spacy_clean'][c3_idx][0] # obama
c3_2 = df['spacy_clean'][c3_idx][5] # scandal
ai_pattern(3,c3_1,c3_2)

## Case 4: President Donald Trump & $1million & Bahama
c4_idx = df[(df['id']==4)&(df['sourceid']==1)]['spacy_clean'].index[0]
df['spacy_clean'][c4_idx]
c4_1 = df['spacy_clean'][c4_idx][2] # trump
c4_2 = df['spacy_clean'][c4_idx][7] # bahama
ai_pattern(4,c4_1,c4_2)

## Case 5: => nothing found with biden & fauci
c5_idx = df[(df['id']==5)&(df['sourceid']==1)]['spacy_clean'].index[0]
df['spacy_clean'][c5_idx]
c5_1 = df['spacy_clean'][c5_idx][3] # biden
c5_2 = df['spacy_clean'][c5_idx][7] # fauci
ai_pattern(5,c5_1,c5_2)

## Case6: Biden & sex offender Jeffrey Epstein => nothing found
c6_idx = df[(df['id']==6)&(df['sourceid']==1)]['spacy_clean'].index[0]
df['spacy_clean'][c6_idx]
c6_1 = df['spacy_clean'][c6_idx][3] # biden
c6_2 = df['spacy_clean'][c6_idx][8] # epstein
ai_pattern(6,c6_1,c6_2)

# df['url'][c6_idx]
'''
## TODO
# AC Key word selection: 1) human name: e.g., politician. 2) location 3) topic
# Data Collection

# Find AI pattern for all pairwises of AC in the claim of fake news
all_patterns = {}
id_max = max(df['id'])
for id_num in range(1,id_max+1):
    all_patterns['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==1)]['spacy_clean'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns['id%s' %id_num].append(ai_pattern(id_num,words[i],words[j]))

print(all_patterns['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['spacy_clean'].reset_index(drop=True)[0]

# Make a CSV file and List of AB & BC patterns.
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/ai_patterns_title.csv"

all_patterns.items()

header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']

with open(output_csv,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2]),case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)
