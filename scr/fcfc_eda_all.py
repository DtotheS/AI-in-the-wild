import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import re
import csv
import ast
os.getcwd()

## plt setting
plt.rc('font', size=14)
plt.style.use('seaborn-bright')

# Snopes data load
sn = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_091922.csv")
sn = sn[sn['page_type']=="Fact Check"] # Only select Fact Checks
sn['date_published']=pd.to_datetime(sn['date_published'])
sn['date_updated']=pd.to_datetime(sn['date_updated'])

sn['yearp'] = sn.apply(lambda row : row['date_published'].year, axis=1)
sn['monthp'] = sn.apply(lambda row : row['date_published'].month, axis=1)
sn['dayp'] = sn.apply(lambda row : row['date_published'].day, axis=1)

sn = sn[sn['date_published'].between(dt(2019,5,17),dt(2022,8,31))]
len(sn) # 5934

sn = sn[sn['rating'].isnull()==False] # 8 mssing ratings. (checked urls. There were no ratings.)
len(sn) # 5933

sn.rename(columns={'url':'link','date_published':"fc_date",'yearp':'fc_year','monthp':'fc_month','dayp':'fc_day','author_name':'author'},inplace=True)
sn.columns

set(sn['rating'].to_list())
sn.loc[sn['rating']=="True","rating"] = "TRUE"
sn.loc[sn['rating']=="False","rating"] = "FALSE"
sn.loc[sn['rating']=="none"] = None

# Politifact data load
pf = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv6_16to22.csv")
len(pf) #10710
pf['fc_date']=pd.to_datetime(pf['fc_date'])
pf['cdate']=pd.to_datetime(pf['cdate'])
pf = pf[pf['fc_date'].between(dt(2019,5,17),dt(2022,8,31))]
len(pf) #5806
pf.isnull().sum()


# AAP data load
aap = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
aap.columns
len(aap) #843
aap['fc_date'] = pd.to_datetime(aap['fc_date'])
aap = aap[aap['fc_date'].between(dt(2019,5,17),dt(2022,8,31))] # 827
aap.isnull().sum() # sum links are missing summarized claims.

# AAP has multiple authors. split it. deliminater = ;
import re
for i in range(len(aap)):
    if aap['author'].iloc[i] == "AAP FactCheck" or aap['author'].iloc[i] == "FactCheck":
        aap['author'].iloc[i] = "AAP Factcheck"
    try:
        foo = re.split("and |& |,", aap['author'].iloc[i])
        aap['author'].iloc[i] = "; ".join(x.strip() for x in foo)
    except:
        pass


# Logically data load
lg = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/logically_090622_v2.csv")
lg = lg[lg['location'] == "United States"] # only for us
len(lg) #993

# Date

def date_img(dataset):
    dname = []
    cname = []
    for yy in range(2019, 2023):
        for mm in range(1, 13):
            dname.append("{}-{}".format(yy, mm))
            print("{}-{}".format(yy, mm))
            try:
                cname.append(dataset.groupby(['fc_year', 'fc_month']).count()['link'][yy][mm])
            except:
                cname.append(0)
    return dname, cname

sndate, sncnt = date_img(sn)
pfdate, pfcnt = date_img(pf)
aapdate, aapcnt = date_img(aap)
lgdate, lgcnt = date_img(lg)
sndate = sndate[4:44]
sncnt = sncnt[4:44]
pfcnt = pfcnt[4:44]
aapcnt = aapcnt[4:44]
lgcnt = lgcnt[4:44]

plt.plot(sncnt,linestyle="-", marker=".", label = "Snopes (Total Fcs: %s)" %len(sn))
plt.plot(pfcnt,linestyle="--", marker="o", label = "Politifact (Total Fcs: %s)" %len(pf))
plt.plot(aapcnt,linestyle="-.", marker="v", label = "AAP (Total Fcs: %s)" %len(aap))
plt.plot(lgcnt,linestyle=":", marker="^", label = "Logically (Total Fcs: %s)" %len(lg))
plt.grid()
plt.legend(fontsize=9)
plt.xticks(np.arange(0,len(sncnt),5), sndate[::5], rotation=90)
plt.subplots_adjust(bottom=0.3, top=0.95, right=0.95)
# plt.title("# FCs for each month (2019-5 ~ 2022-08)")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("Number of articles")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/all/fcs_month_all.png",dpi=600)
plt.show()
plt.close()


# authors
years_li = pf.groupby('fc_year').count().index.to_list()

def author_year(df):
    y = []
    for yy in years_li:
        print(str(yy) + ": # authors = " + str(
            len(df.groupby(['fc_year', 'author']).count().sort_values(['fc_year', 'link'], ascending=False).loc[yy])))
        y.append(
            len(df.groupby(['fc_year', 'author']).count().sort_values(['fc_year', 'link'], ascending=False).loc[yy]))
    x = [str(yy) for yy in years_li]
    return x,y

snx,sny = author_year(sn)
pfx,pfy = author_year(pf)
# aapx,aapy = author_year(aap)
x_axis = np.arange(len(snx))

## Note: after lg subsetting only US, no 2019 articles. Thus, manually added 0 for 2019.
years_li = lg.groupby('fc_year').count().index.to_list()
lgx,lgy = author_year(lg)
lgy.insert(0,0)
lgx[0]='2019'

## revert years_li to 2019 ~ 2022
years_li = pf.groupby('fc_year').count().index.to_list()

for yy in years_li:
    foo = aap[aap['fc_year'] == yy]['author'].to_list()
    foo2 = []
    for i in range(len(foo)):
        try:
            foo2.extend(foo[i].split(";"))
        except:
            foo2.append(foo[i])
    print(set(foo2))
    print(len(set(foo2)))

aapx = years_li
aapy = [14,5,4,16]

aap_auli = []
for i in range(len(aap)):
    try:
        aap_auli.extend([x.strip() for x in aap['author'].iloc[i].split(";")])
    except:
        aap_auli.append(aap['author'].iloc[i])


plt.bar(x_axis - 0.3,sny,width=0.2,label="Snopes"+"(Total: %s)"%len(set(sn['author'])))
plt.bar(x_axis - 0.1,pfy,width=0.2,label="Politifact"+"(Total: %s)"%len(set(pf['author'])))
plt.bar(x_axis + 0.1,aapy,width=0.2,label="AAP"+"(Total: %s)"%len(set(aap_auli)))
plt.bar(x_axis + 0.3,lgy,width=0.2,label="Logically"+"(Total: %s)"%len(set(lg['author'])))
# for i in range(len(x)):
#     plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(snx)),snx,rotation=90)
plt.legend(fontsize=9)
plt.subplots_adjust(bottom=0.2, top=0.95)
# plt.title("# authors by year")
plt.ylabel("Number of authors ")
plt.xlabel("Year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/all/authors_year_all.png",dpi=600)
plt.show()
plt.close()

# Rating

set(sn['rating'])
set(pf['rating'])
set(lg['rating'])
set(aap['rating'])

sn['rating2'] = None
for i in range(len(sn)):
    if sn['rating'].iloc[i] == "FALSE":
        sn['rating2'].iloc[i] = "False"
    elif sn['rating'].iloc[i] == "Mostly False":
        sn['rating2'].iloc[i] = "Mostly False"
    elif sn['rating'].iloc[i] == "Mixture":
        sn['rating2'].iloc[i] = "Mixture"
    elif sn['rating'].iloc[i] == "Mostly True":
        sn['rating2'].iloc[i] = "Mostly True"
    elif sn['rating'].iloc[i] == "TRUE":
        sn['rating2'].iloc[i] = "True"
    else:
        sn['rating2'].iloc[i] = "Others"

pf['rating2'] = None
for i in range(len(pf)):
    if pf['rating'].iloc[i] in ("false","pants-fire"):
        pf['rating2'].iloc[i] = "False"
    elif pf['rating'].iloc[i] == "barely-true":
        pf['rating2'].iloc[i] = "Mostly False"
    elif pf['rating'].iloc[i] == "half-true":
        pf['rating2'].iloc[i] = "Mixture"
    elif pf['rating'].iloc[i] == "mostly-true":
        pf['rating2'].iloc[i] = "Mostly True"
    elif pf['rating'].iloc[i] == "true":
        pf['rating2'].iloc[i] = "True"
    else:
        pf['rating2'].iloc[i] = "Others"
set(pf['rating2'])


lg['rating2'] = None
for i in range(len(lg)):
    if lg['rating'].iloc[i] == "FALSE":
        lg['rating2'].iloc[i] = "False"
    elif lg['rating'].iloc[i] == "MISLEADING":
        lg['rating2'].iloc[i] = "Mostly False"
    elif lg['rating'].iloc[i] == "half-true":
        lg['rating2'].iloc[i] = "Mixture"
    elif lg['rating'].iloc[i] == "PARTLY TRUE":
        lg['rating2'].iloc[i] = "Mostly True"
    elif lg['rating'].iloc[i] == "TRUE":
        lg['rating2'].iloc[i] = "True"
    else:
        lg['rating2'].iloc[i] = "Others"
set(lg['rating2'])

aap['rating2'] = None
for i in range(len(aap)):
    if aap['rating'].iloc[i] == "false":
        aap['rating2'].iloc[i] = "False"
    elif aap['rating'].iloc[i] == "mostly false":
        aap['rating2'].iloc[i] = "Mostly False"
    elif aap['rating'].iloc[i] == "mixture":
        aap['rating2'].iloc[i] = "Mixture"
    elif aap['rating'].iloc[i] == "mostly true":
        aap['rating2'].iloc[i] = "Mostly True"
    elif aap['rating'].iloc[i] == "true":
        aap['rating2'].iloc[i] = "True"
    else:
        aap['rating2'].iloc[i] = "Others"
set(aap['rating2'])
'''
aap[aap['rating'] == "ambiguous"]['link'] # Ambiguous – It is not possible to determine the veracity of the claim.
aap[aap['rating'] == "mixture"]['link']
aap[aap['rating'] == "misleading"]['link']
aap[aap['rating'] == "somewhat false"]['link'] # Somewhat False – The claim has a problem or inaccuracy but it does contain a significant element or elements of truth.
aap[aap['rating'] == "somewhat true"]['link']
'''

def rating_img(df):
    x = ["False", "Mostly False", "Mixture", "Mostly True", "True", "Others"]
    y = [(df['rating2'] == z).sum() for z in x]
    return x,y

snx,sny = rating_img(sn)
pfx,pfy = rating_img(pf)
aapx,aapy = rating_img(aap)
lgx,lgy = rating_img(lg)
x_axis = np.arange(len(snx))

plt.bar(x_axis - 0.3, sny, width=0.2, label="Snopes" + "(Total: %s)" % len(sn))
plt.bar(x_axis - 0.1, pfy, width=0.2, label="Politifact" + "(Total: %s)" % len(pf))
plt.bar(x_axis + 0.1, aapy, width=0.2, label="AAP" + "(Total: %s)" % len(aap))
plt.bar(x_axis + 0.3, lgy, width=0.2, label="Logically" + "(Total: %s)" % len(lg))
plt.xticks(np.arange(len(snx)), snx, rotation=90)
plt.legend()
plt.subplots_adjust(bottom=0.35, top=0.95, left=0.15)
# plt.title("# authors by year")
plt.ylabel("Number of articles")
plt.xlabel("Rating")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/all/rating_all.png",dpi=600)
plt.show()
plt.close()

# Topic handling

## convert string of list into list

## sn: Primary_category
import math
for i in range(len(sn)):
    try:
        if math.isnan(sn['primary_category'].iloc[i]):
            sn['primary_category'].iloc[i] = None
    except:
        pass
sn['primary_category'].isnull().sum() # 6 have float(nan)

## pf: tags
for i in range(len(pf)):
    pf['tags'].iloc[i] = ast.literal_eval(pf['tags'].iloc[i])
pf['tags'].apply(lambda x : None if (len(x) == 0) else x)
len(pf) - pf['tags'].isnull().sum() # all 5806 articles have at least one tag

## lg: key_words
lg.rename(columns={'key_word': 'tags'}, inplace=True)
lg['tags'] = lg['tags'].apply(lambda x: x.strip("'[]`").split("\\n")) #make list as list.
lg['tags'] = lg['tags'].apply(lambda x : None if (x == ['']) else x) #[''] = None
len(lg) - lg['tags'].isnull().sum() # Among 993 articles, 260 has at leas on keyword

def topic_extract(df):
    ks = []
    for i in range(len(df)):
        try:
            ks.extend(df['tags'].iloc[i])
        except:
            pass
    stripString = "`'/"
    ks = [re.sub("[" + stripString + "]", "", key).lower() for key in ks]
    return ks

pf_tags = topic_extract(pf)
len(pf_tags) # total 21531 tags
len(set(pf_tags)) # 1112 unique tags
len(pf_tags) /len(pf) # avg 3.71 tags per article

lg_tags = topic_extract(lg)
len(lg_tags) # total 379 tags
len(set(lg_tags)) # 93 unique tags
len(lg_tags) /len(lg) # avg 0.38 tags per article

sn_tags = sn['primary_category'].to_list()
sn_tags = [x for x in sn_tags if x != "none"]
sn_tags = [re.sub("&amp;","and",x) for x in sn_tags if x]
sn_tags = [x.lower() for x in sn_tags]

set(sn_tags)
len(sn_tags) # 5921 primary topics
len(set(sn_tags)) # 60 unique topics
len(sn_tags) /len(sn) # avg 1 tags per article
len(sn) - len(sn_tags) # 12 does not have primary category

def topic_img(df,ks,num_topic):
    import collections
    counter = collections.Counter(ks)
    sort_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))

    x = list(sort_counter.keys())[:num_topic]
    y = list(sort_counter.values())[:num_topic]

    yp = []
    for i in y:
        yp.append(str(int(round(i / len(df) * 100, 0))) + "%")

    plt.bar(x, y, label="Topic Frequency - Top %s" % num_topic)
    for i in range(len(x)):
        plt.text(x[i], y[i], yp[i], ha='center')
    plt.xticks(np.arange(len(x)), x, rotation=90)
    plt.subplots_adjust(bottom=0.55, top=0.90, right = 0.95, left = 0.2)
    # plt.title("LG: Key Words - Only top 20")
    plt.ylabel("Number of articles (Total: %s)" % (len(df)),loc="top")
    plt.xlabel("Top 10 topics")
    plt.ylim([0,3500])
    # plt.ylim([0, 1200]) # for top author topics
    plt.rc('font', size=14)
    # plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/keyword_freq.png",dpi=600)
    plt.show()
    plt.close()

topic_img(pf,pf_tags,10)
topic_img(sn,sn_tags,10)
topic_img(lg,lg_tags,10)

# Top10 Authors
import string
author_li = ["Author " + x for x in string.ascii_uppercase[0:10]]
sny = sn.groupby('author').count().sort_values(['link'],ascending=False)['link'].to_list()[0:10]
pfy = pf.groupby('author').count().sort_values(['link'],ascending=False)['link'].to_list()[0:10]
aapy = aap.groupby('author').count().sort_values(['link'],ascending=False)['link'].to_list()[0:10]
lgy = lg.groupby('author').count().sort_values(['link'],ascending=False)['link'].to_list()[0:10]
x_axis = np.arange(len(author_li))

plt.bar(x_axis - 0.3, sny, width=0.2, label="Snopes" + "(Total: %s)" % len(sn))
plt.bar(x_axis - 0.1, pfy, width=0.2, label="Politifact" + "(Total: %s)" % len(pf))
plt.bar(x_axis + 0.1, aapy, width=0.2, label="AAP" + "(Total: %s)" % len(aap))
plt.bar(x_axis + 0.3, lgy, width=0.2, label="Logically" + "(Total: %s)" % len(lg))
plt.xticks(np.arange(len(author_li)), author_li, rotation=90)
plt.legend()
plt.subplots_adjust(bottom=0.3, top=0.95, left=0.15, right=0.95)
# plt.title("# authors by year")
plt.ylabel("Number of articles")
plt.xlabel("The 10 most prolific authors")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/all/author10_all.png",dpi=600)
plt.show()
plt.close()

# Ratings of Top 10 authors
def author_dic(df,author_num):
    rating_li = ['False', 'Mostly False', 'Mixture', 'Mostly True', 'True', 'Others']
    y = df.groupby(['author']).count().sort_values(['link'], ascending=False)['link'][:author_num].to_list()
    x = df.groupby(['author']).count().sort_values(['link'], ascending=False)[:author_num].index.tolist()

    df_aurating = pd.DataFrame()
    df_aurating['dummy'] = range(6)
    for nn in x:
        df_aurating['%s' % nn] = pd.Series(
            df.groupby(['author', 'rating2']).count().sort_values(['author', 'link'], ascending=False).loc[nn]['link'][
            :].index.get_level_values('rating2'))
        df_aurating['%s_v' % nn] = pd.Series(
            df.groupby(['author', 'rating2']).count().sort_values(['author', 'link'], ascending=False).loc[nn]['link'][
            :].array)  # since this is Series, .array is needed to add as column in df.
        df_aurating['%s_p' % nn] = pd.Series((df.groupby(['author', 'rating2']).count().sort_values(['author', 'link'],
                                                                                                    ascending=False).loc[
                                                  nn]['link'][:].array /
                                              df.groupby(['author', 'rating2']).count().sort_values(['author', 'link'],
                                                                                                    ascending=False).loc[
                                                  nn]['link'][:].array.sum()) * 100)

    dic_rat = {}
    for name in x:
        dic_rat[name] = []
        for rat in rating_li:
            dic_rat[name].append(df[(df['author'] == name) & (df['rating2'] == rat)].count()[0])

    for name in x:
        dic_rat[name + "_p"] = []
        for rat in rating_li:
            dic_rat[name + "_p"].append(round((df[(df['author'] == name) & (df['rating2'] == rat)].count()[0]) / (
            df[df['author'] == name].count()[0]) * 100, 1))
    return dic_rat

sn_dic = author_dic(sn,10)
pf_dic = author_dic(pf,10)
aap_dic = author_dic(aap,10)
lg_dic = author_dic(lg,10)

def ar_img_per(df_dic):
    # Create a color palette
    palette = plt.get_cmap('Paired')

    import string
    author_li2 = ["Author " + x for x in string.ascii_uppercase[0:10]]
    author_li = list(df_dic.keys())[0:10]
    rating_li = ['False', 'Mostly False', 'Mixture', 'Mostly True', 'True', 'Others']
    # PercentageL top 10 authors FCs by rating
    ## finding: different with politifact, this graph shows that there is not much difference between authors.
    num = 0
    for name in author_li:
        plt.plot(rating_li, df_dic[name + "_p"], marker='', color=palette(num), linewidth=1, alpha=0.9,
                 label=author_li2[num])
        num += 1

    # Add titles
    # plt.title("SN: Relative(%) number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='red')
    plt.xlabel("Rating")
    plt.ylabel("Percentage (%)")
    # Show the graph
    plt.legend(fontsize=8)
    plt.ylim([0, 100])
    plt.xticks(np.arange(len(rating_li)), rating_li, rotation=90)
    plt.subplots_adjust(bottom=0.35, right=0.95, top=0.95)
    # plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_relFcs_rating_author.png",dpi=600)
    plt.show()
    plt.close()


def ar_img_abs(df_dic):
    # Create a color palette
    palette = plt.get_cmap('Paired')

    import string
    author_li2 = ["Author " + x for x in string.ascii_uppercase[0:10]]
    author_li = list(df_dic.keys())[0:10]
    rating_li = ['False', 'Mostly False', 'Mixture', 'Mostly True', 'True', 'Others']
    # PercentageL top 10 authors FCs by rating
    ## finding: different with politifact, this graph shows that there is not much difference between authors.
    num = 0
    for name in author_li:
        plt.plot(rating_li, df_dic[name], marker='', color=palette(num), linewidth=1, alpha=0.9, label=author_li2[num])
        num += 1
    # plt.title("SN: Absolute number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Rating")
    plt.ylabel("Number of articles")
    # Show the graph
    plt.legend(fontsize=8)
    plt.xticks(np.arange(len(rating_li)), rating_li, rotation=90)
    plt.subplots_adjust(bottom=0.35, top=0.90, right=0.95)
    # plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_absFcs_rating_author.png",dpi=600)
    plt.show()
    plt.close()

ar_img_per(sn_dic)
ar_img_abs(lg_dic)

# Topics for the top author
## Sn
sn[sn['author']==list(sn_dic.keys())[0]].groupby('primary_category').count().sort_values(['link'],ascending=False)
ks_sn = sn[sn['author']==list(sn_dic.keys())[0]]['primary_category'].to_list()

## PF
tags_pftop = [x for subli in pf[pf['author']==list(pf_dic.keys())[0]]['tags'].to_list() for x in subli] # The top author of Politifact
## PF top has 3372 tags
len(pf[pf['author']==list(pf_dic.keys())[0]]['link']) ## PF top wrote 1104 articles
from collections import Counter
Counter(tags_pftop).most_common(10)
sum(Counter(tags_pftop).values())

## LG
tags_lgtop = []
for subli in lg[lg['author']==list(lg_dic.keys())[0]]['tags'].to_list():
    try:
        tags_lgtop.extend(subli)
    except:
        pass

lg[lg['author']==list(lg_dic.keys())[0]]['tags'].isnull().sum() # LG top wrote 202 articles, and only 31 articles has tags
len(tags_lgtop) # total 36 tags from 31 articles

from collections import Counter
Counter(tags_lgtop).most_common(10)
sum(Counter(tags_lgtop).values()) # total 36 tags from 31 articles.

ks_pf = topic_extract(pf[pf['author']==list(pf_dic.keys())[0]]) # extract topic for the top1 pf author
topic_img(pf[pf['author']==list(pf_dic.keys())[0]],ks_pf,10)

topic_img(sn[sn['author']==list(sn_dic.keys())[0]],ks_sn,10)

ks_lg = topic_extract(lg[lg['author']==list(lg_dic.keys())[0]]) # extract topic for the top1 lg author
topic_img(lg[lg['author']==list(lg_dic.keys())[0]],ks_lg,10)

# Claim Similarity Comparison
sn.columns
pf.columns
lg.columns
aap.columns

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
from sklearn.metrics.pairwise import cosine_similarity

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