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
# from keybert import KeyBERT
import yake
from nltk.stem import WordNetLemmatizer

## Reference
# Keyword Extraction : https://ourcodeworld.com/articles/read/1613/top-5-best-python-libraries-to-extract-keywords-from-text-automatically

# os.getcwd()
ogdf = pd.read_csv("/AI-in-the-wild/apriori/fc_src.csv") # fc + sources data
fcs = pd.read_csv("/AI-in-the-wild/apriori/factcheck_websites.csv") # 1203 fc data

## For checking labels
label = pd.read_csv("/AI-in-the-wild/apriori/label/ai_label.csv") # sian_label
idlist = label['id'].tolist()
ogdf = ogdf[ogdf['id'].isin(idlist)]
###

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
df['total'] = None
df['sourceid'].astype(int)
df.columns

for i in range(len(df)):
    if df['sourceid'][i] == 0:
        df['total'][i] = df['claim'][i] # Claim for fact check page
    else:
        df['total'][i] = df['title'][i] # Only use title for ai patterns

### OPENIE
# References
# https://github.com/knowitall/openie
# https://github.com/philipperemy/stanford-openie-python
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#api-documentation
# https://stanfordnlp.github.io/CoreNLP/download.html
# https://github.com/smilli/py-corenlp


# !pip install --upgrade pip
# !pip install stanford_openie
# !pip install graphviz
# !pip install pycorenlp

# !pip install stanza #https://stanfordnlp.github.io/stanza/index.html#getting-started
import stanza
# stanza.download('en')
# nlp = stanza.Pipeline('en')
# doc = nlp(df['total'][1])
# print(doc)
# print(doc.entities)

# stanza.install_corenlp()
# !export CORENLP_HOME=~/stanza_corenlp
from stanza.server import CoreNLPClient

df['openie'] = None
with CoreNLPClient(
        annotators=['openie'], #,'tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        output_format='json',
        timeout=30000,
        memory='6G') as client:
    for i in range(len(df)):
        li = []
        if df['sourceid'][i] == 0:
            ann = client.annotate(df['claim'][i],properties={"annotators":"openie","openie.triple.strict":"true","openie.affinity_probability_cap":1/3,"openie.triple.all_nominals":"true"})
            for sen in ann['sentences']:
                for rel in sen['openie']:
                    relationSent = rel['subject'], rel['relation'], rel['object']
                    li.append(relationSent)
            df['openie'][i] = li

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
stops = STOP_WORDS #spacy

# doc=nlp(df['total'][36])
# doc.sentences[0].words[0].lemma.lower().strip() not in stops and doc.sentences[0].words[0].lemma.lower().strip().isalpha()
# test = nlp(df['openie'][0][0][0])
# test.sentences[0].words[0].lemma
# len(df['openie'][0][0])

df['cl_openie'] = None
for t in range(len(df)):
    if df['sourceid'][t] == 0:
        ka_list = []
        kc_list = []
        for i in range(len(df['openie'][t])): # for every triple of relation
            ka = nlp(df['openie'][t][i][0]) # for keyword A
            kc = nlp(df['openie'][t][i][2]) # for keyword C
            for j in range(len(ka.sentences)):
                for k in range(len(ka.sentences[j].words)):
                    foo_worda = ka.sentences[j].words[k].lemma.lower().strip()  # pre-processing: tokenization, lemmatization, lower, strip, remove stop words, special character removal
                    if foo_worda not in stops and foo_worda.isalpha():
                        ka_list.append(str(foo_worda))

            for jj in range(len(kc.sentences)):
                for kk in range(len(kc.sentences[jj].words)):
                    foo_wordc = kc.sentences[jj].words[kk].lemma.lower().strip()  # pre-processing: tokenization, lemmatization, lower, strip, remove stop words, special character removal
                    if foo_wordc not in stops and foo_wordc.isalpha():
                        kc_list.append(str(foo_wordc))
        ka_list = list(set(ka_list))
        kc_list = list(set(kc_list))
        df['cl_openie'][t] = [ka_list,kc_list]

    else:
        doc = nlp(df['total'][t])
        make_li = []
        for j in range(len(doc.sentences)):
            for k in range(len(doc.sentences[j].words)):
                foo_word = doc.sentences[j].words[k].lemma.lower().strip() #pre-processing: tokenization, lemmatization, lower, strip, remove stop words, special character removal
                if foo_word not in stops and foo_word.isalpha():
                    make_li.append(str(foo_word))
        df['cl_openie'][t] = make_li # do not remove duplicates

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

#ai pattern finding algorithm.
def ai_pattern2(c_name,id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)][c_name].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)][c_name].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df[c_name][k1j])
                set2 = set(df[c_name][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    x = Counter(li_commonkeys)
    ordered_x = x.most_common()
    return (k1,k2,ordered_x)

##AI pattern finding using cl_openie
def openie_patterns(cl_name):
    all_patterns = {}
    id_list = df[df['sourceid'] == 0]['id'].tolist()
    # id_max = max(df['id'])
    for id_num in id_list:
        all_patterns['id%s' % id_num] = []
        words = df[(df['id'] == id_num) & (df['sourceid'] == 0)][cl_name].reset_index(drop=True)[0]
        words_lena = len(words[0])
        words_lenb = len(words[1])
        for i in range(words_lena - 1):
            for j in range(i + 1, words_lenb):
                if words[0][i] != words[1][j]:
                    all_patterns['id%s' % id_num].append(ai_pattern2(cl_name, id_num, words[0][i], words[1][j]))
    return all_patterns

pat_openie = openie_patterns('cl_openie')

# openie pattern: Tag True if ai pattern found for each fc for TOTAL
df['openie_pattern'] = False
k = 0
for i in pat_openie:
    for ii in range(len(pat_openie[i])):
        if pat_openie[i][ii][2]>0: # if there exist at least one common 'B' keyword
            df['openie_pattern'][k] = True
            break
    k += 1

def pattern2_words(c_name,id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)][c_name].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)][c_name].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo2]:
                k2_idx.append(foo2)
    # 3. association
    li_commonkeys = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                set1 = set(df[c_name][k1j])
                set2 = set(df[c_name][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                li_commonkeys.extend(inter_list)
    return (k1,k2,li_commonkeys)


df['openie_bwords'] = None
indx = 0
for id in df[df['sourceid']==0]['id']:
    bs = []
    words = df[(df['id'] == id) & (df['sourceid'] == 0)]['cl_openie'].reset_index(drop=True)[0]
    words_lena = len(words[0])
    words_lenb = len(words[1])
    for i in range(words_lena - 1):
        for j in range(i + 1, words_lenb):
            if words[0][i] != words[1][j]:
                bs.extend(pattern2_words('cl_openie',id, words[0][i], words[1][j])[2])
    df['openie_bwords'][indx] = list(set(bs))
    indx += 1

df[df['sourceid']==0]['openie_pattern'].sum() # 58/198, 29.29%
len(df[df['sourceid']==0])

df.to_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df_openie.pkl')
# df = pd.read_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df_openie.pkl')

#ai pattern finding algorithm with titlea, titleb, and claim
def ai_title(c_name,id,key1,key2):
    i = id
    k1_idx = [] # index contains keyword 1
    k2_idx = [] # index contains keyword 2
    k1 = key1
    k2 = key2
    #############################################################################
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==1).
        if df[(df['id']==i)&(df['sourceid']==j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid']==0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news/need to reset_index to change series into datetime and compared the dates.
            foo1 = df[(df['id']==i)&(df['sourceid']==j)][c_name].index[0] #index for id=i & sourceid=j
            if k1 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo1]:
                k1_idx.append(foo1)
    for j in range(1,len(df[df['id']==i])): # not include fake news, but from the first source (sourceid ==2).
        if df[(df['id']==i)&(df['sourceid'] == j)]['datetime'].reset_index(drop=True)[0] <= df[(df['id']==i)&(df['sourceid'] == 0)]['datetime'].reset_index(drop=True)[0]: #datetime of source j <= datetime of the fake news
            foo2 = df[(df['id'] == i) & (df['sourceid'] == j)][c_name].index[0]  # index for id=i & sourceid=j
            if k2 in df[(df['id']==i)&(df['sourceid']==j)][c_name][foo2]:
                k2_idx.append(foo2)
    # 3. association
    ais = []
    for k1j in k1_idx:
        for k2j in k2_idx:
            if k1j != k2j:
                titlea = df['title'][k1j]
                titleb = df['title'][k2j]
                claim = df[df['id']==id]['claim'].reset_index(drop=True)[0]
                set1 = set(df[c_name][k1j])
                set2 = set(df[c_name][k2j])
                inter = set1.intersection(set2)
                inter.discard(k1) # remove keyword A form candidates of B
                inter.discard(k2) # remove keyword C form candidates of B
                inter_list = list(inter)
                numB = len(inter_list)
                ais.append([k1,k2,numB,inter_list,claim,titlea,titleb])
    if len(ais)>0:
        return ais
    else:
        return None

##AI pattern finding using cl_openie, show titlea, titleb and claim.
def openie_title(cl_name):
    all_patterns = {}
    id_list = df[df['sourceid'] == 0]['id'].tolist()
    # id_max = max(df['id'])
    for id_num in id_list:
        all_patterns['id%s' % id_num] = []
        words = df[(df['id'] == id_num) & (df['sourceid'] == 0)][cl_name].reset_index(drop=True)[0]
        words_lena = len(words[0])
        words_lenb = len(words[1])
        for i in range(words_lena - 1):
            for j in range(i + 1, words_lenb):
                if words[0][i] != words[1][j]:
                    foo = ai_title(cl_name, id_num, words[0][i], words[1][j])
                    if foo != None:
                        for tt in range(len(foo)):
                            all_patterns['id%s' % id_num].append(foo[tt])
    return all_patterns

title_openie = openie_title('cl_openie')

# Make a CSV file and List of AB & BC patterns.
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/openie_title198.csv"
def title_csv(path,dict):
    header = ['id','keyword A','keyword C','num_B','keywords B','Claim','Title A','Title C']
    with open(path, 'w') as f:
        c = csv.writer(f)  # write csv on f.
        c.writerow(header)  # header
        for key, value in dict.items():
            for case in value:
                li = []
                li.extend([key, case[0], case[1], case[2],case[3],case[4],case[5],case[6]])  # case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
                c.writerow(li)

title_csv(output_csv,title_openie)

df['openie_title'] = False
k = 0
for i in title_openie:
    for ii in range(len(title_openie[i])):
        if title_openie[i][ii][2]>0: # if there exist at least one common 'B' keyword
            df['openie_title'][k] = True
            break
    k+=1
df['openie_title'].sum()
57/198
#######################################################################
# With labeled data, check the ai-pattern finding performace
df[df['sourceid']==0]['openie_pattern'].sum() # 13/24 54.17%
label.columns
y_pred = df[df['sourceid']==0]['openie_pattern'].tolist()
y_exist = label['label_existence'].tolist()
y_found = label['label_found'].tolist()

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
confusion_matrix(y_exist,y_pred)
accuracye = accuracy_score(y_exist,y_pred)
f1e = f1_score(y_exist,y_pred)
recalle = recall_score(y_exist,y_pred)
pree = precision_score(y_exist,y_pred)
print("existence-label based :", accuracye, f1e, recalle, pree)

y_exist.count(True)
y_pred.count(True)
y_found.count(True)

confusion_matrix(y_found,y_pred)
accuracyf = accuracy_score(y_found, y_pred)
f1f = f1_score(y_found, y_pred)
recallf = recall_score(y_found, y_pred)
pref = precision_score(y_found,y_pred)
print("found-label based :", accuracyf, f1f, recallf, pref)

df[df['sourceid']==0].groupby('legitimacy').count()

# Make a CSV file and List of AB & BC patterns.
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/openie_patterns_198.csv"
def pattern_csv(path,dict):
    header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
    with open(path, 'w') as f:
        c = csv.writer(f)  # write csv on f.
        c.writerow(header)  # header
        for key, value in dict.items():
            for case in value:
                li = []
                li.extend([key, case[0], case[1], bool(case[2]), len(case[2]), " ",
                           case[2]])  # case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
                c.writerow(li)

pattern_csv(output_csv,pat_openie)

y_pred.count(True)
y_exist.count(True)
y_found.count(True)

# lbls = list(set(df['legitimacy'].tolist()))
# for i in lbls:
#     print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern']))*100) + "%)")

# print(all_patterns['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]



# with CoreNLPClient(
#         endpoint="http://localhost:9001",
#         annotators=['openie','tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
#         output_format='json',
#         timeout=30000,
#         memory='6G') as client:
#     # s = "cocaine was found on a cargo ship owned by U.S. Senate Majority Leader and anti-drug politician Mitch McConnell."
#     s = "Some 90 pounds of cocaine was found on a cargo ship owned by U.S. Senate Majority Leader and anti-drug politician Mitch McConnell."
#     # s = df[df['id']==30]['total'][26]
#     output = client.annotate(s,properties={"annotators":"openie","openie.triple.strict":"true","openie.affinity_probability_cap":1/3,"openie.triple.all_nominals":"true"}) #
#     result = [output["sentences"][0]["openie"] for item in output]
#     print(result)
#     for i in result:
#         for rel in i:
#             relationSent = rel['subject'], rel['relation'], rel['object']
#             print(relationSent)
#
# with open('/Users/agathos/DtotheS/AI-in-the-wild/openie.csv','w') as f:
#     header = ['id', 'subject', 'relation', 'object']
#     c = csv.writer(f)
#     c.writerow(header)  # header
#     for i in range(len(df)):
#         if df['openie'][i]:
#             for j in df['openie'][i]:
#                 c.writerow([df['id'][i],j[0],j[1],j[2]])


'''
# Set up classpath: Only when classpath is not working.
!export CLASSPATH="$CLASSPATH:javanlp-core.jar:stanford-corenlp-models-current.jar";
!for file in `find lib -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
# Run server
!java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# check the connection to server
import requests
print(requests.post('http://[::]:9000/?properties={"annotators":"tokenize,ssplit,pos","outputFormat":"json"}', data = {'data':'The quick brown fox jumped over the lazy dog.'}).text)
from pycorenlp import StanfordCoreNLP
nlp=StanfordCoreNLP("http://localhost:9000/")

def openie(sen):
    li = []
    # s = "cocaine was found on a cargo ship owned by U.S. Senate Majority Leader and anti-drug politician Mitch McConnell."
    # s = "Some 90 pounds of cocaine was found on a cargo ship owned by U.S. Senate Majority Leader and anti-drug politician Mitch McConnell."
    s = sen
    output = nlp.annotate(s, properties={"annotators":"tokenize,ssplit,pos,depparse,natlog,openie","outputFormat": "json","triple.strict":"true"})
    result = [output["sentences"][0]["openie"] for item in output]
    # print(result)
    for i in result:
        for rel in i:
            relationSent=rel['subject'],rel['relation'],rel['object']
            # print(relationSent)
            li.append(relationSent)
    return li
'''

## Keyword extraction
# YAKE: https://github.com/LIAAD/yake
# https://amitness.com/keyphrase-extraction/
language = "en"
max_ngram_size = 2
deduplication_thresold = 0.3
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20
stops = STOP_WORDS #spacy

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
lemma = WordNetLemmatizer()

# text = df['total'][0]
# keywords = custom_kw_extractor.extract_keywords(text)
# keywords
# df['ke_total'][]
# df['total'][0]

df['ke12_total'] = np.nan
for i in range(len(df['ke12_total'])):
    foo = keywords = custom_kw_extractor.extract_keywords(df['total'][i])
    make_li = []
    for j in foo:
        # if len(j[0].split())>1:
        make_li.append(lemma.lemmatize(j[0]).lower())
    df['ke12_total'][i] = make_li

df['ke_total'][0]
df['ke2_total'][0]
df['ke12_total'][2]
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
'''
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

def ai_pattern_words(id,key1,key2):
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
    return (k1,k2,li_commonkeys)

#### Find all AI pattern for all pairwises of AC in the claim of fake news BASED ON THE TOTAL
def all_patterns(cl_name):
    all_patterns = {}
    id_list = df[df['sourceid'] == 0]['id'].tolist()
    # id_max = max(df['id'])
    for id_num in id_list:
        all_patterns['id%s' % id_num] = []
        words = df[(df['id'] == id_num) & (df['sourceid'] == 0)][cl_name].reset_index(drop=True)[0]
        words_len = len(words)
        for i in range(words_len - 1):
            for j in range(i + 1, words_len):
                all_patterns['id%s' % id_num].append(ai_pattern2(cl_name, id_num, words[i], words[j]))

    return all_patterns

pat1 = all_patterns('cl_total')

# All ai pattern: Tag True if ai pattern found for each fc for TOTAL
df['ai_pattern'] = np.nan
k = 0
for i in pat1:
    for ii in range(len(pat1[i])):
        if pat1[i][ii][2]:
            df['ai_pattern'][k] = True
            break
    k += 1
df['ai_pattern'].sum() # 111, 56%

# lbls = list(set(df['legitimacy'].tolist()))
# for i in lbls:
#     print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern']))*100) + "%)")

# print(all_patterns['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]

# Make a CSV file and List of AB & BC patterns.
all_patterns.items()
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/patterns_v2.csv"
def pattern_csv(path,dict):
    header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
    with open(path, 'w') as f:
        c = csv.writer(f)  # write csv on f.
        c.writerow(header)  # header
        for key, value in dict.items():
            for case in value:
                li = []
                li.extend([key, case[0], case[1], bool(case[2]), len(case[2]), " ",
                           case[2]])  # case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
                c.writerow(li)

pat1 = all_patterns('cl_total')
pattern_csv(output_csv,pat1)


df['ai_Bwords'] = np.nan
indx = 0
for id in df[df['sourceid']==0]['id']:
    bs = []
    words = df[(df['id'] == id) & (df['sourceid'] == 0)]['cl_total'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            bs.extend(ai_pattern_words(id,words[i],words[j])[2])
    df['ai_Bwords'][indx] = bs
    indx += 1
df['ai_Bwords'][198]

# words = df[(df['id'] == 1) & (df['sourceid'] == 0)]['cl_total'].reset_index(drop=True)[0]
# ai_pattern_words(1,words[11],words[12])[2]
# len(words)


#### NE ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
pat_ne = all_patterns('ne_total')
'''
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
'''
# Check whether there is at least one ai pattern found for each fc(id) for Named ENTITY ai pattern
df['ai_pattern_ent'] = np.nan
k = 0
for i in pat_ne:
    for ii in range(len(pat_ne[i])):
        if pat_ne[i][ii][2]:
            df['ai_pattern_ent'][k] = True
            break
    k += 1
df['ai_pattern_ent'].sum() # not found.

# print(all_patterns2['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]


#### NE and Tokenize ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
pat_netk = all_patterns('netk_total')
'''
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
'''

# Make a CSV file and List of AB & BC patterns.
output_csv3 = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/patterns_ent_tk_v1.csv"
pattern_csv(output_csv3,pat_netk)
'''
header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
with open(output_csv3,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns3.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2])," ",case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)
'''

#### NE only for claim, and Tokenize ai pattern: Find AI pattern for all pairwises of AC in the claim of fake news BASED ON ENTITY#####
def ai_pattern_ent_claim_words(id,key1,key2):
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
    return (k1,k2,li_commonkeys)

pat_netkclaim = all_patterns('netk_claim')
'''
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
'''


output_csv4 = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/words.csv"
pattern_csv(output_csv4,pat_netkclaim)

'''
header = ['id','keyword A','keyword C','exist_B','num_B','keywords B']
with open(output_csv4,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns4.items():
        for case in value:
            li = []
            li.extend([key,case[0],case[1],bool(case[2]),len(case[2])," ",case[2]]) #case[0]=A , case[1]=C, case[2] = Bs, bool(li) False if empty
            c.writerow(li)
'''
df['ai_pattern_ent_claim'] = np.nan
k = 0
for i in pat_netkclaim:
    for ii in range(len(pat_netkclaim[i])):
        if pat_netkclaim[i][ii][2]:
            df['ai_pattern_ent_claim'][k] = True
            break
    k += 1

df['ai_pattern_ent_claim'].sum() # 67, 33.8%

df[df['ai_pattern_ent_claim']==True]['id'].tolist()
# print(all_patterns3['id1'])
# df[(df['id']==3)&(df['sourceid']==1)]['cl_total'].reset_index(drop=True)[0]

## Now, only pick the words.
'''
all_patterns4_words = {}
id_list =df[df['sourceid']==0]['id'].tolist()
# id_max = max(df['id'])
for id_num in id_list:
    all_patterns4_words['id%s' %id_num] = []
    words = df[(df['id']==id_num)&(df['sourceid']==0)]['netk_claim'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            all_patterns4_words['id%s' %id_num].extend(ai_pattern_ent_claim_words(id_num,words[i],words[j])[2])
'''

df['ai_netk_claim_Bwords'] = np.nan
indx = 0
for id in df[df['sourceid']==0]['id']:
    bs = []
    words = df[(df['id'] == id) & (df['sourceid'] == 0)]['netk_claim'].reset_index(drop=True)[0]
    words_len = len(words)
    for i in range(words_len-1):
        for j in range(i+1,words_len):
            bs.extend(ai_pattern_ent_claim_words(id,words[i],words[j])[2])
    df['ai_netk_claim_Bwords'][indx] = bs
    indx += 1

df['ai_Bwords'][0]
df['ai_netk_claim_Bwords'][0]


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
df.to_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df3.pkl')
df = pd.read_pickle('/AI-in-the-wild/apriori/df3.pkl')

#### Apply KE and find ai pattern
pat_ke = all_patterns('ke_total')
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/ke_patterns.csv"
pattern_csv(output_csv,pat_ke)

# All ai pattern: Tag True if ai pattern found for each fc for TOTAL
df['ke_ai_pattern'] = np.nan
k = 0
for i in pat_ke:
    for ii in range(len(pat_ke[i])):
        if pat_ke[i][ii][2]:
            df['ke_ai_pattern'][k] = True
            break
    k += 1
df['ke_ai_pattern'].sum() # 110, 56%

pat_ke2 = all_patterns('ke2_total')
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/ke2_patterns.csv"
pattern_csv(output_csv,pat_ke2)

# All ai pattern: Tag True if ai pattern found for each fc for TOTAL
df['ke2_ai_pattern'] = np.nan
k = 0
for i in pat_ke2:
    for ii in range(len(pat_ke2[i])):
        if pat_ke2[i][ii][2]:
            df['ke2_ai_pattern'][k] = True
            break
    k += 1
df['ke2_ai_pattern'].sum() # 14

# KE with 1 and 2 grams, duplicate threshold = 0.3
pat_ke12 = all_patterns('ke12_total')
output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/ke12_patterns.csv"
pattern_csv(output_csv,pat_ke12)

df['ke12_ai_pattern'] = np.nan
k = 0
for i in pat_ke12:
    for ii in range(len(pat_ke12[i])):
        if pat_ke12[i][ii][2]:
            df['ke12_ai_pattern'][k] = True
            break
    k += 1
df['ke12_ai_pattern'].sum() # 26