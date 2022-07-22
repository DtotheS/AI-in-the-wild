############ EDA ############
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv

os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v3_072122.csv")
len(df) # Total 21262
df = df.rename(columns={df.columns[0]: "id" })
df['id']
df.columns
df.isnull().sum()

df['fc_date']=pd.to_datetime(df['fc_date'])
df['cdate']=pd.to_datetime(df['cdate'])

df['fc_year'] = df.apply(lambda row : row['fc_date'].year, axis=1)
df['fc_month'] = df.apply(lambda row : row['fc_date'].month, axis=1)
df['fc_day'] = df.apply(lambda row : row['fc_date'].day, axis=1)

df['fc_year'].isnull().sum()
df['fc_month'].isnull().sum()
df['fc_day'].isnull().sum()

df = df[df['fc_year'].between(2016,2021)] # Select FCs published between 2016 and 2021
len(df) # total # FCs: 9534
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv3_16to21.csv",index=False)

years_li=list(set(df['fc_year']))
years_li = [int(x) for x in years_li]
years_li.sort()

month_li = list(set(df['fc_month']))
month_li = [int(x) for x in month_li]
month_li.sort()

## rating
rating = df.groupby(['fc_year','rating']).count().sort_values(['fc_year'],ascending=False)#.loc[2016]
ratings_li = list(set(rating.index.get_level_values('rating')))
print(ratings_li)
ratings_li = ['true','mostly-true','half-true','barely-true', 'false','pants-fire'] # Make an order

rating_dic={}
rating_dic['name']=ratings_li
for yy in years_li:
    rating_dic[yy]=[]
    for rr in ratings_li:
        try:
            rating_dic[yy].append(rating.loc[yy].loc[rr][0])
        except:
            rating_dic[yy].append(0)

# x: ratings, y: # FCs, line: years
# Change the style of plot
plt.style.use('seaborn-bright')

for yy in years_li:
    plt.plot(rating_dic['name'], rating_dic[yy], linestyle="-", marker="o", label="%s" %(yy))
plt.legend()
plt.xticks(np.arange(len(rating_dic['name'])), rating_dic['name'], rotation=90)
plt.subplots_adjust(bottom=0.3, top=0.99)
plt.ylabel("number of FCs")
plt.xlabel("ratings")
plt.show()
plt.close()

# x: months, y: #FCs, line: years
count = df.groupby(['fc_year','fc_month']).count().sort_values(['fc_year','fc_month'],ascending=False)
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
plt.ylabel("number of FCs")
plt.xlabel("Month")
plt.show()
plt.close()

# number of FCs: 2016-01 ~ 2021-12
count_li = count['id'].to_list()
count_li.reverse()
monthall = []
for yy in years_li:
    for mm in month_li:
        monthall.append(str(yy)+"-"+str(mm))
plt.plot(monthall, count_li, linestyle="-", marker="o", label="# FCs")
plt.xticks(np.arange(0,len(monthall),5),[monthall[i] for i in np.arange(0,len(monthall),5)],rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.99)
plt.legend()
plt.grid()
plt.ylabel("number of FCs")
plt.xlabel("Year-Month")
plt.show()
plt.close()

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

# Where the misinformation comes from
y = df.groupby('cwhere').count().sort_values('id',ascending=False)[:20]['id'].tolist()
x = df.groupby('cwhere').count().sort_values('id',ascending=False)[:20].index.tolist()

plt.bar(x,y)
addlabels(x,y)
plt.xticks(np.arange(len(x)), x, rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.xlabel("Where the misinformation come from")
plt.ylabel("Frequency (total # FCs = %s)" %len(df))
plt.legend()
plt.show()
plt.close()
