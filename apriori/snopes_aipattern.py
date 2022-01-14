import numpy as np
import pandas as pd
import os
# use spacy for keyword extraction
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from datetime import datetime as dt
import csv
import matplotlib.pyplot as plt
import numpy as np

# os.getcwd()
ogdf = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/fc_src.csv") # fc + sources data
fcs = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/factcheck_websites.csv") # 1203 fc data

## Select ids which only have sources.
fc = ogdf[ogdf['sourceid']==0]
fc['sources_num'] = fc['sources_num'].astype(int)
fc_src = fc[fc['sources_num']>0]['id'] # C1: contains only ids with sources
id_src = fc_src.to_list() # Among 1203, only 198 contains sources
df_src = ogdf['id'].isin(id_src)
df = ogdf[df_src]
# df = df[df['date'] != "none"] #
df = df.reset_index(drop=True)

# Use title and text(claim or content)
df['total'] = np.nan
df['sourceid'].astype(int)
df.columns

for i in range(len(df)):
    if df['sourceid'][i] == 0:
        df['total'][i] = df['claim'][i] # Claim for fact check page
    else:
        df['total'][i] = df['title'][i] # Only use title for ai patterns

## text cleaning.

# nlp = spacy.load("en_core_web_trf") # better accuracy
nlp = spacy.load("en_core_web_sm") # efficient
# stops = stopwords.words("english") # nltk
stops = STOP_WORDS #spacy
'''
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
'''

# Clean data: Lemmatization, remove stop words, lower, strip(), only alphabet
df['cl_total'] = np.nan
for i in range(len(df['cl_total'])):
    foo = nlp(df['total'][i])
    text = ""
    for j in range(len(foo)):
        foo2 = foo[j].lemma_.strip() # Lemmatization, strip()
        if foo2 not in stops and foo2.isalpha(): # remove stop words
            text += foo2+" "
    # df['ne_total'][i] = nlp(text).ents # Named Entity
    df['cl_total'][i] = nlp(text.lower()) # lower
    make_li = []
    for j in range(len(df['cl_total'][i])):
        make_li.append(str(df['cl_total'][i][j]))
    df['cl_total'][i] = make_li
df['cl_total'][0]

# Named Entity Recognition: cl_total + NE applied.
df['ne_total'] = np.nan
for i in range(len(df['ne_total'])):
    foo = nlp(df['total'][i])
    text = ""
    for j in range(len(foo)):
        foo2 = foo[j].lemma_.strip() # Lemmatization, strip()
        if foo2 not in stops and foo2.isalpha(): # remove stop words
            text += foo2+" "
    text = nlp(text).ents # Named Entity
    make_li = []
    for j in text:
        make_li.append(str(j).lower())
    df['ne_total'][i] = make_li
df['ne_total'][]

# NE and Tokenize:
# Named Entity Recognition: cl_total + NE applied.
df['netk_total'] = np.nan
for i in range(len(df['netk_total'])):
    foo = nlp(df['total'][i])
    text = ""
    for j in range(len(foo)):
        foo2 = foo[j].lemma_.strip() # Lemmatization, strip()
        if foo2 not in stops and foo2.isalpha(): # remove stop words
            text += foo2+" "
    text = nlp(text).ents # Named Entity
    text2 = ""
    for jj in text:
        text2 += str(jj)+" "
    text3 = nlp(text2.lower()) # Lower
    make_li = []
    for jjj in text3:
        make_li.append(str(jjj))
    df['netk_total'][i] = make_li



# Only apply NE and Tokenize for Claim.
df['netk_claim'] = np.nan
for i in range(len(df['netk_claim'])):
    foo = nlp(df['total'][i])
    text = ""
    for j in range(len(foo)):
        foo2 = foo[j].lemma_.strip() # Lemmatization, strip()
        if foo2 not in stops and foo2.isalpha(): # remove stop words
            text += foo2+" "
    if df['sourceid'][i]>0: # for source, X NE
        df['netk_claim'][i] = nlp(text.lower()) # lower
        make_li = []
        for j in range(len(df['netk_claim'][i])):
            make_li.append(str(df['netk_claim'][i][j]))
        df['netk_claim'][i] = make_li
    else: # for Claim, O NE
        text = nlp(text).ents  # Named Entity
        text2 = ""
        for jj in text:
            text2 += str(jj) + " "
        text3 = nlp(text2.lower())  # Lower
        make_li = []
        for jjj in text3:
            make_li.append(str(jjj))
        df['netk_claim'][i] = make_li

# for i in range(250):
#     if df['netk_claim'][i] != df['netk_total'][i]:
#         print(i)
# df['netk_claim'][244]


# df['ne_total'][7]
# nlp(df['total'][0]).ents[0]
# df['cl_total'][3]
# type(str(df['cl_total'][0][1]))


# df[df['sourceid']>0]['ne_total'][216] #check 216 vs. 217

# df['total'][20]
# df['ne_total'][20]
# df['ne_total'][0]=nlp(df['total'][0]).ents
# len(df['ne_total'][0])
'''
df['nlp_total'] = np.nan
for i in range(len(df['total'])):
    df['nlp_total'][i] = str(nlp(df['total'][i]))

df['nlp_entity'] = np.nan
for i in range(len(df['total'])):
    df['nlp_entity'][i] = str(nlp(df['total'][i]).ents) # Find named entities, phrases and concepts

df['cl_total'] = df['nlp_total'].apply(normalize, lowercase=True, remove_stopwords=True)    
df['cl_entity'] = df['nlp_entity'].apply(normalize, lowercase=True, remove_stopwords=True)
'''
# better to clean after key extraction
# for the claim, key exraction make too small set, so it is better not to do.


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

# 1. mark sources which does not have datetime.
df['datetime'] = np.nan
for i in range(len(df['date'])):
    try:
        df['datetime'][i] = dt.strptime(df['date'][i],"%m/%d/%y")
    except:
        try:
            df['datetime'][i] = dt.strptime(df['date'][i], "%m/%d/%Y")
        except:
            df['datetime'][i] = dt(3000,12,31) # exclude sources which does not have datetime.

# # select sources before the fake news date
# for i in range(len(df[df['id']==1])):
#     if df['datetime'][i] <= df[(df['sourceid'] == 1) & (df['id'] == 1)]['datetime'][0]:
#         print(df['datetime'][i])

'''
## Removed duplicate keywords for each source.
def spl(row):
    # Split keywords by pace and make a list of words.
    li = row.split()
    # Remove duplicate keywords for each list
    li = list(set(li))
    return li

df['cl_total'] = df['cl_total'].apply(spl)
df['cl_entity'] = df['cl_entity'].apply(spl)

# df['cl_total'][8]
'''

## Find ai in the wild

# 2. key word selection
# k1 = df['cl_total'][0][1] # cocaine
# k2 = df['cl_total'][0][11] # mitch

'''
# 3. find ai pattern for all ids with random key words.
import random

common_keys = { }
for i in range(1,max(df['id'])+1):
    # arbitrary choose keyword the first and second one. Need to be modified #
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    rd = random.sample(range(0,len(df[(df['id']==i)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0])),2) # randomly generate two numbers based on the maximum number of keywords for the claim of the fake news
    k1 = df[(df['id']==i)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0][rd[0]] # df of "id=1 &source=1" will give only one row ([0]) and want to prent the first [0][0] and the second [0][1] keywords for now.
    k2 = df[(df['id']==i)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0][rd[1]]
    #############################################################################
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['cl_total'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['cl_total'][foo1]:
                k1_idx.append(foo1)
    for j in range(2,len(df[df['id']==i])+1): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 1)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['cl_total'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['cl_total'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['cl_total'][k1j])
                set2 = set(df['cl_total'][k2j])
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
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['cl_total'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['cl_total'][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['cl_total'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['cl_total'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['cl_total'][k1j])
                set2 = set(df['cl_total'][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)


def ai_pattern_ent(id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['ne_total'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['ne_total'][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['ne_total'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['ne_total'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['ne_total'][k1j])
                set2 = set(df['ne_total'][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)

def ai_pattern_ent_tk(id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['netk_total'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['netk_total'][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['netk_total'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['netk_total'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['netk_total'][k1j])
                set2 = set(df['netk_total'][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)

def ai_pattern_ent_claim(id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)]['netk_claim'].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)]['netk_claim'][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)]['netk_claim'].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)]['netk_claim'][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df['netk_claim'][k1j])
                set2 = set(df['netk_claim'][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)
'''
# Check sources date which are late then article publsihed date.
id_list =df[df['sourceid']==0]['id'].tolist()
for id in id_list:
    for j in range(1, len(df[df['id'] == id])):  # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id'] == id) & (df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] > df[(df['id'] == id) & (df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]:
            print("id and source id: ", id,j,df[(df['id'] == id) & (df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0])

df[(df['id']==23)&(df['sourceid']==0)]['url']
'''

'''
## Case 1: Cocaine & Ping May & Mitch
c1_1 = df['cl_total'][0][1] # cocaine
c1_2 = df['cl_total'][0][11] # mitch
ai_pattern(1,c1_1,c1_2)

## Case 2: Trump & Obama & School Lunch Program
c2_1 = df['cl_total'][8][1] # trump
c2_2 = df['cl_total'][8][5] # school
ai_pattern(2,c2_1,c2_2)

## Case 3: obama & Gordon Ernst & college admissions bribery scandal
c3_idx = df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].index[0]
c3_1 = df['cl_total'][c3_idx][0] # obama
c3_2 = df['cl_total'][c3_idx][5] # scandal
ai_pattern(3,c3_1,c3_2)

## Case 4: President Donald Trump & $1million & Bahama
c4_idx = df[(df['id']==4)&(df['sourceid']==1)]['cl_total'].index[0]
df['cl_total'][c4_idx]
c4_1 = df['cl_total'][c4_idx][2] # trump
c4_2 = df['cl_total'][c4_idx][7] # bahama
ai_pattern(4,c4_1,c4_2)

## Case 5: => nothing found with biden & fauci
c5_idx = df[(df['id']==5)&(df['sourceid']==1)]['cl_total'].index[0]
df['cl_total'][c5_idx]
c5_1 = df['cl_total'][c5_idx][3] # biden
c5_2 = df['cl_total'][c5_idx][7] # fauci
ai_pattern(5,c5_1,c5_2)

## Case6: Biden & sex offender Jeffrey Epstein => nothing found
c6_idx = df[(df['id']==6)&(df['sourceid']==1)]['cl_total'].index[0]
df['cl_total'][c6_idx]
c6_1 = df['cl_total'][c6_idx][3] # biden
c6_2 = df['cl_total'][c6_idx][8] # epstein
ai_pattern(6,c6_1,c6_2)

# df['url'][c6_idx]
'''
## TODO
# AC Key word selection: 1) human name: e.g., politician. 2) location 3) topic
# Data Collection

#### Find all AI pattern for all pairwises of AC in the claim of fake news BASED ON THE TOTAL
all_patterns = {}
id_list =df[df['sourceid']==0]['id'].tolist()
# id_max = max(df['id'])
for id_num in id_list:
    all_patterns['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==0)]['cl_total'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns['id%s' %id_num].append(ai_pattern(id_num,words[i],words[j]))

# print(all_patterns['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]

# Make a CSV file and List of AB & BC patterns.
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/patterns_v2.csv"

all_patterns.items()

header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']

with open(output_csv,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2])," ",case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)

# All ai pattern: Tag True if ai pattern found for each fc for TOTAL
df['ai_pattern'] = np.nan
k = 0
for i in all_patterns:
    for ii in range(len(all_patterns[i])):
        if all_patterns[i][ii][2]:
            df['ai_pattern'][k] = True
            break
    k += 1

df['ai_pattern'].sum() # 111, 56%

# lbls = list(set(df['legitimacy'].tolist()))
# for i in lbls:
#     print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern']))*100) + "%)")

#### NE ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
all_patterns2 = {}
id_list =df[df['sourceid']==0]['id'].tolist()
# id_max = max(df['id'])
for id_num in id_list:
    all_patterns2['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==0)]['ne_total'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns2['id%s' %id_num].append(ai_pattern_ent(id_num,words[i],words[j]))

# Check whether there is at least one ai pattern found for each fc(id) for Named ENTITY ai pattern
df['ai_pattern_ent'] = np.nan
k = 0
for i in all_patterns2:
    for ii in range(len(all_patterns2[i])):
        if all_patterns2[i][ii][2]:
            df['ai_pattern_ent'][k] = True
            break
    k += 1
df['ai_pattern_ent'].sum() # not found.

# print(all_patterns2['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]


#### NE and Tokenize ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
all_patterns3 = {}
id_list =df[df['sourceid']==0]['id'].tolist()
# id_max = max(df['id'])
for id_num in id_list:
    all_patterns3['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==0)]['netk_total'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns3['id%s' %id_num].append(ai_pattern_ent_tk(id_num,words[i],words[j]))

df['ai_pattern_ent_tk'] = np.nan
k = 0
for i in all_patterns3:
    for ii in range(len(all_patterns3[i])):
        if all_patterns3[i][ii][2]:
            df['ai_pattern_ent_tk'][k] = True
            break
    k += 1
df['ai_pattern_ent_tk'].sum() #41, 20.7%


# Make a CSV file and List of AB & BC patterns.
output_csv3 = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/patterns_ent_tk_v1.csv"

header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
with open(output_csv3,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns3.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2])," ",case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)


#### NE and Tokenize ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
all_patterns4 = {}
id_list =df[df['sourceid']==0]['id'].tolist()
# id_max = max(df['id'])
for id_num in id_list:
    all_patterns4['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==0)]['netk_claim'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns4['id%s' %id_num].append(ai_pattern_ent_claim(id_num,words[i],words[j]))

df['ai_pattern_ent_claim'] = np.nan
k = 0
for i in all_patterns4:
    for ii in range(len(all_patterns4[i])):
        if all_patterns4[i][ii][2]:
            df['ai_pattern_ent_claim'][k] = True
            break
    k += 1

df['ai_pattern_ent_claim'].sum() # 67, 33.8%
# print(all_patterns3['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]

# Make a CSV file and List of AB & BC patterns.
output_csv4 = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/patterns_ent_tk_claim.csv"
# all_patterns4.items()
header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
with open(output_csv4,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns4.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2])," ",case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)

# Check whether there is at least one ai pattern found for each fc(id) for Named ENTITY ai pattern
# Calculate delta day
df['delta_dt'] = np.nan
for i in range(len(df)):
    if df['datetime'][i] < dt(3000,1,1): # only for there is date time.
        delta = df[(df['id'] == df['id'][i]) & (df['sourceid'] == 0)]['datetime'].iloc[0] - df['datetime'][i] # FC - current dt: positive means past, negative means future based on FC.
        df['delta_dt'][i] = delta.days

len(df[df['delta_dt'].isna()]) # Total 31 nan = no dates
from datetime import datetime
# Remove sources longer than today. ### Need to fix this later....
today = datetime.now()
today
for i in range(len(df)):
    if df['datetime'][i] > today:
        print('id, sourceid, and datetime: ', df['id'][i],df['sourceid'][i],df['datetime'][i],df['url'][i])
        df['delta_dt'][i] = np.nan
len(df[df['delta_dt'].isna()])  # became 38: There are 7 cases which shows wrong date for source. We removed those.

# Safe df for EDA later.
# dfnew = pd.read_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df2.pkl')
df.to_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df2.pkl')
