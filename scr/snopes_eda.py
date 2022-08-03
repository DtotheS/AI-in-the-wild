import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/fcs_16to21.csv")
df.columns
len(df) # Total 11073

df = df[df['page_type']=="Fact Check"] # Only select Fact Checks
len(df) # 11068

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

df = df[df['yearp'].between(2016,2021)] # Select FCs published between 2016 and 2021
len(df) # total # FCs: 10679
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_16to21.csv",index=False)

years_li=list(set(df['yearp']))
years_li = [int(x) for x in years_li]
years_li.sort()

month_li = list(set(df['monthp']))
month_li = [int(x) for x in month_li]
month_li.sort()

#rating
rating = df.groupby(['yearp','rating']).count().sort_values(['yearp','id'],ascending=False)#.loc[2016]
ratings_li = list(set(rating.index.get_level_values('rating')))

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
for yy in [2017,2018,2019]:
    plt.plot(rating_dic['name'], rating_dic[yy], linestyle="-", marker="o", label="%s" %(yy))
plt.legend()
plt.xticks(np.arange(len(rating_dic['name'])), rating_dic['name'], rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.show()
plt.close()

# number of FCs: years comparison
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
plt.xticks(np.arange(len(count_dic['name'])), count_dic['name'], rotation=90)
# plt.subplots_adjust(bottom=0.4, top=0.99)
plt.show()

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
plt.show()

# number of FCs: year comparison
cnt_years=[]
for yy in years_li:
    cnt_years.append(count.loc[yy].sum()[0])
plt.bar(years_li, cnt_years,label="# FCs")
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
# plt.subplots_adjust(bottom=0.2, top=0.99)
plt.legend()
plt.show()
plt.close()

# Aggregate the years
allrating = []
for rr in ratings_li:
    allrating.append(df.groupby('rating').count()['id'][rr])
plt.bar(ratings_li, allrating,label="# FCs")
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.legend()
plt.show()
plt.close()


# Category
category = df.groupby(['yearp','primary_category']).count().sort_values(['yearp','id'],ascending=False)

df_category = pd.DataFrame()
df_category['name'] = category.index.get_level_values('primary_category')
# df_category['2016'] = category.loc[2016][:20].index.get_level_values('primary_category')
# df_category['2016_v'] = category.loc[2016][:20]['id'].array
# type(category.loc[2016][:20]['id'])
# type(category.loc[2016][:20].index.get_level_values('primary_category'))
for yy in years_li:
    df_category['%d' % yy] = category.loc[yy][:20].index.get_level_values('primary_category')
    df_category['%d_v' % yy] = category.loc[yy][:20]['id'].array # since this is Series, .array is needed to add as column in df.
    df_category['%d_p' % yy] = (category.loc[yy][:20]['id'].array/category.loc[yy][:20]['id'].array.sum())*100

df_category.to_csv("./AI-in-the-wild/data/df_category.csv",index=True)

df.groupby('primary_category').count().sort_values(['yearp','id'],ascending=False).index.get_level_values
cnt_cat=[]
for cc in df.groupby('primary_category').count().sort_values(['yearp','id'],ascending=False).index:
    cnt_cat.append(df.groupby('primary_category').count().sort_values(['yearp','id'],ascending=False).loc[cc]['id'])
x=df.groupby('primary_category').count().sort_values(['yearp','id'],ascending=False).index
y=df.groupby('primary_category').count().sort_values(['yearp','id'],ascending=False)['id']
plt.bar(x[:30], y[:30],label="# FCs")
plt.xticks(np.arange(len(x[:30])),x[:30],rotation=90)
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.legend()
plt.show()
plt.close()

# Author
len(set(df['author_name'])) # There are only 18 authors in total
len(df)/len(set(df['author_name'])) # 593.27 FCs per person during 6 years

author_dic={}
df.groupby('author_name').count().describe()
author=df.groupby(['yearp','author_name']).count().sort_values(['yearp','id'],ascending=False)
author_dic['name']=list(set(df['author_name']))

# FCs/author
cnt_aufcs=[]
for nn in author_dic['name']:
    cnt_aufcs.append(df.groupby('author_name').count().loc[nn]['id'])
plt.bar(author_dic['name'], cnt_aufcs,label="# FCs")
plt.xticks(np.arange(len(author_dic['name'])),author_dic['name'],rotation=90)
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.legend()
plt.show()
plt.close()
# df.groupby('author_name').count().loc['Alex Kasprak']['id']

# number of authors in each year
for year in years_li:
    print("# authors in %s:" %(year),len(author.loc[year]))

for year in years_li:
    print("FCS/author in %s:" % year, round(np.array(count_dic[year]).sum()/len(author.loc[year]),1))

#Author by Topic
df_author = pd.DataFrame()
for nn in author_dic['name']:
    df_author['%s' % nn] = pd.Series(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10].index.get_level_values('primary_category'))
    df_author['%s_v' % nn] = pd.Series(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10].array)  # since this is Series, .array is needed to add as column in df.
    df_author['%s_p' % nn] = pd.Series((df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10].array / df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10].array.sum()) * 100)
    # print(df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10])
# df.groupby(['author_name','primary_category']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:10].index.get_level_values('primary_category')

df_author.to_csv("./AI-in-the-wild/data/df_author.csv",index=True)

#Author by Rating
df_aurating = pd.DataFrame()
df_aurating['dummy'] = range(20)
for nn in author_dic['name']:
    df_aurating['%s' % nn] = pd.Series(df.groupby(['author_name','rating']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:].index.get_level_values('rating'))
    df_aurating['%s_v' % nn] = pd.Series(df.groupby(['author_name','rating']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:].array)  # since this is Series, .array is needed to add as column in df.
    df_aurating['%s_p' % nn] = pd.Series((df.groupby(['author_name','rating']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:].array / df.groupby(['author_name','rating']).count().sort_values(['author_name','id'],ascending=False).loc[nn]['id'][:].array.sum()) * 100)

df_aurating.to_csv("./AI-in-the-wild/data/df_aurating.csv",index=True)

#tags

len(df[df['sources_num']>0]) # 0 == 4588 0> 6091

