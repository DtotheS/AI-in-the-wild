import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_091922.csv")
df.columns
len(df) # Total 15492

df = df[df['page_type']=="Fact Check"] # Only select Fact Checks
len(df) # 15483

df.isnull().sum()

df['date_published']=pd.to_datetime(df['date_published'])
# dt.strptime(df["date_published"][0],'%d-%b-%y')
df['date_updated']=pd.to_datetime(df['date_updated'])

df['yearp'] = df.apply(lambda row : row['date_published'].year, axis=1)
df['monthp'] = df.apply(lambda row : row['date_published'].month, axis=1)
df['dayp'] = df.apply(lambda row : row['date_published'].day, axis=1)

df['yearp'].isnull().sum()
df['dayp'].isnull().sum()
df['monthp'].isnull().sum()

df = df[df['date_published'].between(dt(2016,1,1),dt(2022,8,31))] # select data: 2016.1.1 < 2022.8.31
len(df) # 11633

df = df[df['rating'].isnull()==False] # 8 mssing ratings. (checked urls. There were no ratings.)
len(df) # 11625
df.isnull().sum()

years_li=list(set(df['yearp']))
years_li = [int(x) for x in years_li]
years_li.sort()

month_li = list(set(df['monthp']))
month_li = [int(x) for x in month_li]
month_li.sort()

# Ratings
set(df['rating'].to_list())

df.loc[df['rating']=="True","rating"] = "TRUE"
df.loc[df['rating']=="False","rating"] = "FALSE"
df.loc[df['rating']=="none"] = None

plt.style.use('seaborn-bright')

df = df[df['date_published'].between(dt(2019,5,17),dt(2022,8,31))]
# y: # FCS, x: Dates (for each month)
df.groupby(['yearp','monthp']).count()['url']
dname = []
dcnt = []
for yy in range(2019,2023):
    for mm in range(1,13):
        dname.append("{}-{}".format(yy,mm))
        print("{}-{}".format(yy,mm))
        try:
            dcnt.append(df.groupby(['yearp', 'monthp']).count()['url'][yy][mm])
        except:
            dcnt.append(0)
len(dname) == len(dcnt)
len(dname)
dname = dname[4:44]
dcnt = dcnt[4:44] # ~ 2022-7

plt.plot(dcnt,linestyle="-", marker="o")
plt.xticks(np.arange(0,len(dcnt),5), dname[::5], rotation=90)
plt.grid()
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Snopes: # FCs for each month (2019-5 ~ 2022-08)")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("number of FCs "+"(Total # FCs: %s)" %len(df))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_month.png",dpi=600)
plt.show()
plt.close()

# y: FCs, x: ratings
x = df.groupby(['rating']).count()['url'].index.to_list()
y = df.groupby(['rating']).count()['url'].to_list()
yp = []
for i in y:
    yp.append(str(int(round(i/len(df) * 100,0)))+"%")

plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("SN: # FCs by Rating")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Ratings")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_rating.png",dpi=600)
plt.show()
plt.close()

#rating
rating = df.groupby(['yearp','rating']).count().sort_values(['yearp'],ascending=False)#.loc[2016]
ratings_li = df.groupby(['rating']).count()['url'].index.to_list()
ratings_li = ['FALSE',
'Mostly False',
'Mixture',
'Mostly True',
'TRUE',
 'Correct Attribution',
 'Labeled Satire',
 'Legend',
 'Legit',
 'Misattributed',
 'Miscaptioned',
 'No Rating',
 'Originated as Satire',
 'Outdated',
 'Recall',
 'Research In Progress',
 'Scam',
 'Unfounded',
 'Unproven']

rating_dic={}
rating_dic['name']=ratings_li
for yy in years_li:
    rating_dic[yy]=[]
    for rr in ratings_li:
        try:
            rating_dic[yy].append(rating.loc[yy].loc[rr][0])
        except:
            rating_dic[yy].append(0)

# for yy in years_li:
for yy in [2019,2020,2021,2022]:
    plt.plot(rating_dic['name'], rating_dic[yy], linestyle="-", marker="o", label="%s" %(yy))
plt.legend()
plt.xticks(np.arange(len(rating_dic['name'])), rating_dic['name'], rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.title("# FCs by Rating & Year")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Ratings")
# plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/fcs_rating_years.png",dpi=600)
plt.show()
plt.close()

# number of FCs: years comparison
'''Color cycle to original 7 deafult sylte
import matplotlib.style
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])import matplotlib.style
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
'''
count = df.groupby(['yearp','monthp']).count().sort_values(['yearp','monthp'],ascending=False)
count_dic={}
count_dic['name']=month_li
for yy in years_li:
    count_dic[yy]=[]
    for mm in month_li:
        try:
            count_dic[yy].append(count.loc[yy].loc[mm][0])
        except:
            count_dic[yy].append(0)

for yy in years_li:
    plt.plot(count_dic['name'], count_dic[yy], linestyle="-", marker="o", label="%s" %(yy))
plt.legend()
plt.xticks(np.arange(1,len(count_dic['name'])+1), count_dic['name'], rotation=90)
# plt.subplots_adjust(bottom=0.4, top=0.99)
plt.title("# FCs by Month & Year")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Month")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/fcs_nonth_years.png",dpi=600)
plt.show()


# number of FCs: year comparison
cnt_years=[]
for yy in years_li:
    cnt_years.append(count.loc[yy].sum()[0])
plt.bar(years_li, cnt_years,label="# FCs")
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
# plt.subplots_adjust(bottom=0.2, top=0.99)
plt.legend()
plt.title("# FCs by Year")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/fcs_years.png",dpi=600)
plt.show()
plt.close()

# Category
category = df.groupby(['yearp','primary_category']).count().sort_values(['yearp','url'],ascending=False)

df_category = pd.DataFrame()
df_category['name'] = category[:20].index.get_level_values('primary_category')
# df_category['2016'] = category.loc[2016][:20].index.get_level_values('primary_category')
# df_category['2016_v'] = category.loc[2016][:20]['url'].array
# type(category.loc[2016][:20]['url'])
# type(category.loc[2016][:20].index.get_level_values('primary_category'))
for yy in years_li:
    df_category['%d' % yy] = category.loc[yy][:20].index.get_level_values('primary_category')
    df_category['%d_v' % yy] = category.loc[yy][:20]['url'].array # since this is Series, .array is needed to add as column in df.
    df_category['%d_p' % yy] = (category.loc[yy][:20]['url'].array/category.loc[yy][:20]['url'].array.sum())*100

df.groupby('primary_category').count().sort_values(['yearp','url'],ascending=False).index.get_level_values
cnt_cat=[]
for cc in df.groupby('primary_category').count().sort_values(['yearp','url'],ascending=False).index:
    cnt_cat.append(df.groupby('primary_category').count().sort_values(['yearp','url'],ascending=False).loc[cc]['url'])
x=df.groupby('primary_category').count().sort_values(['yearp','url'],ascending=False).index
y=df.groupby('primary_category').count().sort_values(['yearp','url'],ascending=False)['url']
yp = []
for i in y:
    yp.append(str(int(round(i/len(df) * 100,0)))+"%")
plt.bar(x[:20], y[:20],label="# FCs")
for i in range(len(x[:20])):
    plt.text(x[i], y[i], yp[i], ha = 'center')

plt.xticks(np.arange(len(x[:20])),x[:20],rotation=90)
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
plt.subplots_adjust(bottom=0.4, top=0.9)
plt.legend()
plt.title("SN: # FCs by Primary Category - Only top 20")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Primary Category" + "(Total # categories: %s)"%len(x))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_pcategory.png",dpi=600)
plt.show()
plt.close()

# Author
len(set(df['author_name'])) # There are only 18 authors in total
len(df)/len(set(df['author_name'])) # 645.83 FCs per person during 6 years

author_dic={}
df.groupby('author_name').count().describe()
author=df.groupby(['yearp','author_name']).count().sort_values(['yearp','url'],ascending=False)
author_dic['name']=list(set(df['author_name']))

# FCs/author
x = df.groupby('author_name').count().sort_values(['url'],ascending=False).index.tolist()
y = df.groupby('author_name').count().sort_values(['url'],ascending=False)['url'].to_list()

yp = []
for i in y:
    yp.append(str(int(round(i/len(df) * 100,1)))+"%")

plt.bar(x, y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
plt.subplots_adjust(bottom=0.4, top=0.90)
# plt.legend()
plt.title("SN: # FCs by Authors")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Authors" +"(Total # Authors: %s)"%len(author_dic['name']))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_author.png",dpi=600)
plt.show()
plt.close()
# df.groupby('author_name').count().loc['Alex Kasprak']['url']


# authors for each year
years_li = df.groupby('yearp').count().index.to_list()
y = []
for yy in years_li:
    print(str(yy) +": # authors = "+str(len(df.groupby(['yearp','author_name']).count().sort_values(['yearp','url'],ascending=False).loc[yy])))
    y.append(len(df.groupby(['yearp','author_name']).count().sort_values(['yearp','url'],ascending=False).loc[yy]))
x = [str(int(yy)) for yy in years_li]

plt.bar(x,y,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("SN: # authors by year")
plt.ylabel("number of authors "+"(Total: %s)"%len(set(df['author_name'])))
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_authors_year.png",dpi=600)
plt.show()
plt.close()

# average fc/author for each year
y2 = df.groupby('yearp').count()['url'].to_list() # number of FCs per year
for i in range(len(y)): # here, y is defined by # author per year. from the above code.
    print(str(x[i])+"'s FCs per Author: "+str(round(y2[i]/y[i],2)))
    y2[i] = round(y2[i]/y[i],2)

plt.bar(x,y2,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y2[i], y2[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("SN: FCs/Authors by year")
plt.ylabel("Number of FCs per author")
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_perauthor.png",dpi=600)
plt.show()
plt.close()


#Author by Rating
y = df.groupby(['author_name']).count().sort_values(['url'],ascending=False)['url'][:].to_list()
x = df.groupby(['author_name']).count().sort_values(['url'],ascending=False)[:].index.tolist()

df_aurating = pd.DataFrame()
df_aurating['dummy'] = range(20)
for nn in x:
    df_aurating['%s' % nn] = pd.Series(df.groupby(['author_name','rating']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:].index.get_level_values('rating'))
    df_aurating['%s_v' % nn] = pd.Series(df.groupby(['author_name','rating']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:].array)  # since this is Series, .array is needed to add as column in df.
    df_aurating['%s_p' % nn] = pd.Series((df.groupby(['author_name','rating']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:].array / df.groupby(['author_name','rating']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:].array.sum()) * 100)

ratings_li = ['TRUE','Mostly True','Mixture','Mostly False','FALSE',
 'Correct Attribution',
 'Labeled Satire',
 'Legend',
 'Legit',
 'Misattributed',
 'Miscaptioned',
 'No Rating',
 'Originated as Satire',
 'Outdated',
 'Recall',
 'Research In Progress',
 'Scam',
 'Unfounded',
 'Unproven']

dic_rat = {}
for name in x:
    dic_rat[name] = []
    for rat in ratings_li:
        dic_rat[name].append(df[(df['author_name']==name) & (df['rating']==rat)].count()[0])

for name in x:
    dic_rat[name+"_p"] = []
    for rat in ratings_li:
        dic_rat[name+"_p"].append(round((df[(df['author_name']==name) & (df['rating']==rat)].count()[0])/(df[df['author_name']==name].count()[0])*100,1))

# Create a color palette
palette = plt.get_cmap('Paired')

## top 10 authors FCs by rating
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("SN: Absolute number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Rating")
plt.ylabel("# FCs")
# Show the graph
plt.legend()
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_absFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

# PercentageL top 10 authors FCs by rating
## finding: different with politifact, this graph shows that there is not much difference between authors.
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name+"_p"],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("SN: Relative(%) number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Ratings")
plt.ylabel("Percentage (%)")
# Show the graph
plt.legend()
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.35, top=0.90)
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_relFcs_rating_author.png",dpi=600)
plt.show()
plt.close()


#Author by Topic
x = df.groupby(['author_name']).count().sort_values(['url'],ascending=False)[:].index.tolist()
x
df_author = pd.DataFrame()
for nn in x:
    df_author['%s' % nn] = pd.Series(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10].index.get_level_values('primary_category'))
    df_author['%s_v' % nn] = pd.Series(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10].array)  # since this is Series, .array is needed to add as column in df.
    df_author['%s_p' % nn] = pd.Series((df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10].array / df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10].array.sum()) * 100)
    # print(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10])
# df.groupby(['author_name','primary_category']).count().sort_values(['author_name','url'],ascending=False).loc[nn]['url'][:10].index.get_level_values('primary_category')

for name in x:
    x2 = df_author[df_author[name].notnull()][name].to_list()
    y2 = df_author[df_author[name+"_v"].notnull()][name+"_v"].to_list()
    y2p = df_author[df_author[name + "_p"].notnull()][name + "_p"].to_list()
    tot = df.groupby(['author_name']).count().sort_values(['url'],ascending=False)[:]['url'][name]
    plt.bar(x2,y2, label="# FCs")
    for i in range(len(x2)):
        plt.text(x2[i], y2[i], round(y2p[i],1), ha='center')
    plt.xticks(np.arange(len(x2)), x2, rotation=90)
    plt.subplots_adjust(bottom=0.4, top=0.90)
    plt.title("SN: %s's top 10 primary categories" %name)
    plt.ylabel("Number of FCs" + "(Total # FCs = %s)"%tot)
    plt.xlabel("Primary Category")
    plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_fcs_pcat_%s.png"%name, dpi=600)
    plt.show()
    plt.close()

# Sources Domains
import ast
sdf = df[df['sources_num']>0] # 6847
sdf.columns
sdf['sources'] = df['sources'].apply(lambda x: ast.literal_eval(x))
sdf['sources'].iloc[0]

import re # to extract url from source
import tldextract # to extract domain name from each url

dms = []
urls = []
i=0
link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
for src in sdf['sources']:
    for k in range(len(src)):
        links = re.findall(link_regex,src[k]) # need to be updated. performance getting link is not that good.
        print(len(links))
        i+=1
        for lnk in links:
            # print(lnk[0])
            urls.append(lnk[0])
            ext = tldextract.extract(lnk[0])  # extract domain name
            # print(ext.domain)
            dms.append(ext.domain)

len(dms) # 5739
len(set(dms)) # 1747

import collections
counter = collections.Counter(dms)
sort_counter = dict(sorted(counter.items(), key=lambda item: item[1],reverse=True))

x = list(sort_counter.keys())[:20]
y = list(sort_counter.values())[:20]

yp = []
for i in y:
    yp.append(str(int(round(i/len(df) * 100,1)))+"%")

plt.bar(x,y,label="Sources & References - Top 20")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.3, top=0.90)
plt.title("SN: Sources & References - Top 20")
plt.ylabel("Frequency"+"(Total frequency: %s)"%len(dms))
plt.xlabel("Source Name (Total # keywords: %s)" %len(set(dms)))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/snopes/s_source_freq.png",dpi=600)
plt.show()
plt.close()
