############ EDA ############
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv6_16to22.csv")
len(df) #10710
df['fc_date']=pd.to_datetime(df['fc_date'])
df['cdate']=pd.to_datetime(df['cdate'])
df = df[df['fc_date'].between(dt(2016,1,1),dt(2022,8,31))]
len(df) #10710
df2 = df[df['fc_date'].between(dt(2019,5,17),dt(2022,8,31))] # overlap period for all the 4 FC websites
len(df2) # 5806

years_li=list(set(df['fc_year']))
years_li = [int(x) for x in years_li]
years_li.sort()

month_li = list(set(df['fc_month']))
month_li = [int(x) for x in month_li]
month_li.sort()

plt.style.use('seaborn-bright')

# y: # FCS, x: Dates (for each month)
# Finding: peak (8) reaches early compared to other websites (11).
df.groupby(['fc_year','fc_month']).count()['link']
dname = []
dcnt = []
for yy in range(2016,2023):
    for mm in range(1,13):
        dname.append("{}-{}".format(yy,mm))
        print("{}-{}".format(yy,mm))
        try:
            dcnt.append(df.groupby(['fc_year', 'fc_month']).count()['link'][yy][mm])
        except:
            dcnt.append(0)
len(dname) == len(dcnt)

ol_dname = dname[40:80] # Only for OL peridos
ol_dcnt = dcnt[40:80]
plt.plot(ol_dcnt,linestyle="-", marker="o")
plt.grid()
plt.xticks(np.arange(0,len(ol_dcnt),5), ol_dname[::5], rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Politifact: # FCs for each month (2019-5 ~ 2022-08)")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("number of FCs "+"(Total # FCs: %s)" %len(df2))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_fcs_month.png",dpi=600)
plt.show()
plt.close()

dname[:80]
dcnt[:80]
plt.plot(dcnt[:80],linestyle="-", marker="o")
plt.grid()
plt.xticks(np.arange(0,len(dcnt[:80]),5), dname[:80][::5], rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Politifact: # FCs for each month (2016-1 ~ 2022-08)")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("number of FCs "+"(Total # FCs: %s)" %len(df))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_fcs_month2.png",dpi=600)
plt.show()
plt.close()

# y: FCs, x: ratings
x = df2.groupby(['rating']).count()['link'].index.to_list()
x = [x[-1], x[2], x[3], x[0], x[1], x[4]]
y = df2.groupby(['rating']).count()['link'].to_list()
y = [y[-1], y[2], y[3], y[0], y[1], y[4]]
yp = []
for i in y:
    yp.append(str(round(i/len(df2) * 100,2))+"%")

plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Politifact: # FCs by Rating")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df2))
plt.xlabel("Ratings")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_fcs_rating.png",dpi=600)
plt.show()
plt.close()
'''
# number of FCs: x: years
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i], y[i], y[i], ha = 'center')

cnt_years=[]
for yy in years_li:
    cnt_years.append(count.loc[yy].sum()[0])
plt.bar(years_li, cnt_years,label="# FCs")
addlabels(years_li,cnt_years)
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
# plt.subplots_adjust(bottom=0.2, top=0.99)
plt.legend()
plt.ylabel("number of FCs")
plt.xlabel("Year")
plt.show()
plt.close()

# rating: Aggregate the years
allrating = []
for rr in ratings_li:
    allrating.append(df.groupby('rating').count()['id'][rr])
plt.bar(ratings_li, allrating,label="# FCs")
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.3, top=0.99)
addlabels(ratings_li,allrating)
plt.legend()
plt.ylabel("number of FCs")
plt.xlabel("ratings")
plt.show()
plt.close()
'''

# y: FCs, x: Authors
x = df2.groupby('author').count().index.to_list()
y = df2.groupby('author').count()['link'].to_list()
for a, b in zip(x,y):
    print(a,b)

## only want to show top 20 authors, cause there are too many!
dic_author = {}
for i in range(len(x)):
    dic_author[x[i]]=y[i]

dic_author = dict(sorted(dic_author.items(), key=lambda item: item[1],reverse=True))

x = list(dic_author.keys())[:20]
y = list(dic_author.values())[:20]
yp = []
for i in y:
    yp.append(str(int(round(i/len(df2) * 100,0)))+"%")

# FCS for each author - only top 20 authors
plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("PF: # FCs by author - Top 20 only")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df2))
plt.xlabel("Author")
# plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_fcs_20author.png",dpi=600)
plt.show()
plt.close()


# authors for each year
years_li = df2.groupby('fc_year').count().index.to_list()
y = []
for yy in years_li:
    print(str(yy) +": # authors = "+str(len(df2.groupby(['fc_year','author']).count().sort_values(['fc_year','link'],ascending=False).loc[yy])))
    y.append(len(df2.groupby(['fc_year','author']).count().sort_values(['fc_year','link'],ascending=False).loc[yy]))
x = [str(yy) for yy in years_li]

plt.bar(x,y,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("PF: # authors by year")
plt.ylabel("number of authors "+"(Total: %s)"%len(set(df2['author'])))
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_authors_year.png",dpi=600)
plt.show()
plt.close()

# average fc/author for each year
y2 = df2.groupby('fc_year').count()['link'].to_list() # number of FCs per year
for i in range(len(y)): # here, y is defined by # author per year. from the above code.
    print(str(x[i])+"'s FCs per Author: "+str(round(y2[i]/y[i],2)))
    y2[i] = round(y2[i]/y[i],2)

plt.bar(x,y2,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y2[i], y2[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("PF: FCs/Authors by year")
plt.ylabel("Number of FCs per author")
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_fcs_perauthor.png",dpi=600)
plt.show()
plt.close()

# Ratings of top20 authors
y = df2.groupby(['author']).count().sort_values(['link'],ascending=False)['link'][:20].to_list()
x = df2.groupby(['author']).count().sort_values(['link'],ascending=False)[:20].index.tolist()

df2_aurating = pd.DataFrame()
df2_aurating['dummy'] = range(len(set(df2['rating'])))
for nn in x:
    df2_aurating['%s' % nn] = pd.Series(df2.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].index.get_level_values('rating'))
    df2_aurating['%s_v' % nn] = pd.Series(df2.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array)  # since this is Series, .array is needed to add as column in df2.
    df2_aurating['%s_p' % nn] = pd.Series((df2.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array / df2.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array.sum()) * 100)

ratings_li = ['true', 'half-true', 'mostly-true', 'barely-true', 'false', 'pants-fire']
dic_rat = {}
for name in x:
    dic_rat[name] = []
    for rat in ratings_li:
        dic_rat[name].append(df2[(df2['author']==name) & (df2['rating']==rat)].count()[0])

for name in x:
    dic_rat[name+"_p"] = []
    for rat in ratings_li:
        dic_rat[name+"_p"].append(round((df2[(df2['author']==name) & (df2['rating']==rat)].count()[0])/(df2[df2['author']==name].count()[0])*100,1))


# Create a color palette
palette = plt.get_cmap('Paired')

## top 10 authors FCs by rating
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("PF: Absolute number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Rating")
plt.ylabel("# FCs")
# Show the graph
plt.legend()
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_absFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

# PercentageL top 10 authors FCs by rating
## finding: different with politifact, this graph shows that there is not much difference between authors.
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name+"_p"],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("PF: Relative(%) number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Ratings")
plt.ylabel("Percentage (%)")
# Show the graph
plt.legend()
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/politifact/s_relFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

'''
# Author
df['author'][6119] = "Unknown"
len(set(df['author'])) # There are 356 authors in total
len(df)/len(set(df['author'])) # 26.79 FCs per person during 6 years

author_dic={}
df.groupby('author').count().describe()
author=df.groupby(['fc_year','author']).count().sort_values(['fc_year','id'],ascending=False)
names = [x[1] for x in author.index]
author_dic['name']=sorted(set(names),key=names.index) #to remove duplicates and preserve order

# FCs/author
cnt_aufcs=[]
for nn in author_dic['name']:
    cnt_aufcs.append(df.groupby('author').count().loc[nn]['id'])

# Top 10 most FCs authors
tt = df.groupby(['author']).count().sort_values(['id'],ascending=False)['id'].sum()
aa = len(df.groupby(['author']).count().index.to_list())
y = df.groupby(['author']).count().sort_values(['id'],ascending=False)['id'][:10].to_list()
x = df.groupby(['author']).count().sort_values(['id'],ascending=False)[:10].index.tolist()
plt.bar(x,y,label="# FCs")
addlabels(x,y)
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.ylabel("number of FCs "+"("+str(tt)+")")
plt.xlabel("top 10 author " +"("+str(aa)+")")
plt.legend()
plt.show()
plt.close()

# Top 10 most FCs authors per year
for year in years_li:
    tt = df[df['fc_year']==year].groupby(['fc_year','author']).count().sort_values(['id'],ascending=False)['id'].sum() #Total # FCs for each year
    aa = len(df[df['fc_year']==year].groupby(['fc_year','author']).count().index.to_list())
    foo = df[df['fc_year']==year].groupby(['fc_year','author']).count().sort_values(['id'],ascending=False)['id'][:10]
    x = [x[1] for x in foo.index]
    y = foo.to_list()
    plt.bar(x,y)
    addlabels(x,y)
    plt.xticks(np.arange(len(x)),x,rotation=90)
    plt.subplots_adjust(bottom=0.4, top=0.90)
    plt.ylim(0,400)
    plt.xlabel(str(year)+" authors "+"("+str(aa)+")")
    plt.ylabel("# FCs"+"("+str(tt)+")")
    plt.legend()
    plt.show()
    plt.close()

# # Fcs, authors, average for each year
for year in years_li:
    fcs = df.groupby(['fc_year']).get_group(year).count()[0]
    aus = len(df.groupby(['fc_year']).get_group(year).groupby('author').count())
    aver = round(fcs/aus,2)
    print(str(year) +" FCs: " + str(fcs))
    print(str(year) + " Authors: " + str(aus))
    print(str(year) +" FCs/Authors: " + str(aver))

# Ratings of top10 authors
y = df.groupby(['author']).count().sort_values(['id'],ascending=False)['id'][:10].to_list()
x = df.groupby(['author']).count().sort_values(['id'],ascending=False)[:10].index.tolist()

df_aurating = pd.DataFrame()
df_aurating['dummy'] = range(6)
for nn in x:
    df_aurating['%s' % nn] = pd.Series(df.groupby(['author','rating']).count().sort_values(['author','id'],ascending=False).loc[nn]['id'][:].index.get_level_values('rating'))
    df_aurating['%s_v' % nn] = pd.Series(df.groupby(['author','rating']).count().sort_values(['author','id'],ascending=False).loc[nn]['id'][:].array)  # since this is Series, .array is needed to add as column in df.
    df_aurating['%s_p' % nn] = pd.Series((df.groupby(['author','rating']).count().sort_values(['author','id'],ascending=False).loc[nn]['id'][:].array / df.groupby(['author','rating']).count().sort_values(['author','id'],ascending=False).loc[nn]['id'][:].array.sum()) * 100)

# df_aurating.to_csv("./AI-in-the-wild/data/pf_author_ratings.csv",index=True)

dic_rat = {}
for name in x:
    dic_rat[name] = []
    for rat in ratings_li:
        dic_rat[name].append(df[(df['author']==name) & (df['rating']==rat)].count()[0])

for name in x:
    dic_rat[name+"_p"] = []
    for rat in ratings_li:
        dic_rat[name+"_p"].append(round((df[(df['author']==name) & (df['rating']==rat)].count()[0])/(df[df['author']==name].count()[0])*100,1))


# Create a color palette
palette = plt.get_cmap('Paired')

# # Fcs
num = 0
for name in x:
    num+=1
    plt.plot(ratings_li,dic_rat[name],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("# FCs for each rating", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Ratings")
plt.ylabel("# FCs")
# Show the graph
plt.legend()
plt.show()
plt.close()

# Percentage
num = 0
for name in x:
    num+=1
    plt.plot(ratings_li,dic_rat[name+"_p"],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("# of each rating for each author", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Ratings")
plt.ylabel("Percentage (%)")
# Show the graph
plt.legend()
plt.show()
plt.close()
'''

# Where the misinformation comes from --- Need to be improved.
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(x[i], y[i], y[i], ha = 'center')
y = df.groupby('cwhere').count().sort_values('link',ascending=False)[:20]['link'].tolist()
x = df.groupby('cwhere').count().sort_values('link',ascending=False)[:20].index.tolist()

plt.bar(x,y)
addlabels(x,y)
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.xlabel("Where the misinformation come from")
plt.ylabel("Frequency (total # FCs = %s)" %len(df))
plt.legend()
plt.show()
plt.close()

import ast
for i in range(len(df)):
    x = df['tags'][i]
    x = ast.literal_eval(x)  # list of sources
    x = [n.strip() for n in x]
    df['tags'][i] = x

tag_li = df['tags'].tolist()

from collections import Counter
tags = Counter(t for clist in tag_li for t in clist)
top20 = tags.most_common(20)
sum(tags.values()) / len(df) # 36612 tags from 9534: 3.84 tags

# Top 20 Tags
x = []
y = []
for i in range(len(top20)):
    x.append(top20[i][0])
    y.append(top20[i][1])
plt.bar(x,y)
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.show()

## Modules
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Start to Collect Data
'''
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

driver.get("https://www.politifact.com/")
len(driver.find_elements_by_css_selector("ul.m-togglist__list")[0].find_elements_by_tag_name("a"))

issues = []
driver.get("https://www.politifact.com/issues/")
for i in range(len(driver.find_elements_by_class_name("c-chyron__value"))):
    iss = driver.find_elements_by_class_name("c-chyron__value")[i].text
    issues.append(iss)


persons = []
driver.get("https://www.politifact.com/personalities/")
for i in range(len(driver.find_elements_by_class_name("c-chyron__value"))):
    pers = driver.find_elements_by_class_name("c-chyron__value")[i].text
    persons.append(pers)


import pickle
with open("/Users/agathos/DtotheS/AI-in-the-wild/data/pf_persons", "wb") as ff:
    pickle.dump(persons, ff)
with open("/Users/agathos/DtotheS/AI-in-the-wild/data/pf_issues", "wb") as ff:
    pickle.dump(issues, ff)
'''
import pickle
with open("/Users/agathos/DtotheS/AI-in-the-wild/data/pf_persons", "rb") as ff:   # Unpickling
    persons = pickle.load(ff)

with open("/Users/agathos/DtotheS/AI-in-the-wild/data/pf_issues", "rb") as ff:   # Unpickling
    issues = pickle.load(ff)