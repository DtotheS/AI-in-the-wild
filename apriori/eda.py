import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

src = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/fc_src.csv") # fc + sources data
fc = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/factcheck_websites.csv") # 1203 fc data

len(src) # Total = 2271
len(fc) #FC = 1203 => src = 2271 - 1203 = 1068
len(fc[fc['title']=="none"]['id'])
len(fc[fc['url']=="none"]['id'])
len(src[src['title']=="none"]['id']) # Used "" to collect title. Among 1068 only 99 were not collected. But, among collected, there may be dummy titles.
len(src[src['url']=="none"]['id']) #Among 1068, 718 does not contains url. That is mainly because they did not contain urls before 12/21/2019

fc.columns
x = fc['legitimacy'].value_counts().index.tolist()
y = fc['legitimacy'].value_counts().tolist()
plt.bar(x,y)
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Ratings for 1,203 fact checks')
plt.ylabel('Number of fact checks')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rating_dist_1203.png',dpi=600)
plt.show()
plt.close()
# x = fc.groupby('legitimacy').count()['id'].index.to_list()
# y = fc.groupby('legitimacy').count()['id'].to_list()

## Legitimacy distribution.
fc['legitimacy'].value_counts()
fc['legitimacy'].value_counts(normalize=True)

# fc[fc['legitimacy']=="none"]["url"]

## Legitimacy distribution for 198.
fc[fc['sources_num']>0]['legitimacy'].value_counts()
fc[fc['sources_num']>0]['legitimacy'].value_counts().sum() #total 198 which contains at least one source
fc[fc['sources_num']>0]['legitimacy'].value_counts().plot(kind='bar')
plt.show()
fc[fc['sources_num']>0]['legitimacy'].value_counts(normalize=True)

## Sources Distribution
fc[fc['sources_num']>0]['sources_num'].value_counts().plot(kind='bar')
fc[fc['sources_num']>0]['sources_num'].value_counts().describe()
plt.show()
fc[fc['sources_num']>0]['sources_num'].value_counts()
fc[fc['sources_num']>0]['sources_num'].value_counts(normalize=True)
fc['sources_num'].value_counts(normalize=True)

## Sources distribution grouped by legitimacy (for 1,203 total)

fc[['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False)
fc[['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).plot(kind='bar')
plt.show()
fc[['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False)
# num per article
norm_src = fc[['legitimacy', 'sources_num']].groupby('legitimacy').sum() / fc[['legitimacy','sources_num']].groupby('legitimacy').count()
norm_src.sort_values('sources_num',ascending=False)

## Sources distribution grouped by legitimacy (for 198 contains at least one source)
fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False)
#percentage
fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False)/fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sum()
fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).plot(kind='bar')
plt.show()
#num per article
norm_src = fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum() / fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').count()
norm_src.sort_values('sources_num',ascending=False)

## Real vs Fake
fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).iloc[[0,3]].sum() #real
fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).iloc[[1,6]].sum() #fake
fake = fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).iloc[[0,3]].sum()/fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').count().sort_values('sources_num',ascending=False).iloc[[0,3]].sum()
real = fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').sum().sort_values('sources_num',ascending=False).iloc[[1,6]].sum()/fc[fc['sources_num']>0][['legitimacy','sources_num']].groupby('legitimacy').count().sort_values('sources_num',ascending=False).iloc[[1,6]].sum()
fake, real

### Load df which include delta_dt (time gap days), ai_pattern (all ai), ai_pattern_ent (NE ai)
df = pd.read_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df.pkl')

## EDA
# Among all ai: 58.6%
df['ai_pattern'].sum()
df[df['sourceid']==0]['ai_pattern'].sum() / len(df[df['sourceid']==0])# among 198, 116 have ai pattern.

# Among NE ai: 20.7%
df['ai_pattern_ent'].sum()
df[df['sourceid']==0]['ai_pattern_ent'].sum() / len(df[df['sourceid']==0])# among 198, only 41 have ne ai pattern.

## Find percentage of existence of ai pattern for each label.
# All AI pattern
lbls = list(set(df['legitimacy'].tolist()))
for i in lbls:
    print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern']))*100) + "%)")

# NE AI Pattern
lbls = list(set(df['legitimacy'].tolist()))
for i in lbls:
    print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent']))*100) + "%)")

# df['rating'] match id and its rating for all sources.
df['rating'] = np.nan
for i in range(len(df)):
    df['rating'][i] = df.loc[(df['sourceid']==0)&(df['id'] == df['id'][i])]['legitimacy'].reset_index(drop=True)[0]

df['src_ai_pattern'] = np.nan
for i in range(len(df)):
    df['src_ai_pattern'][i] = df.loc[(df['sourceid']==0)&(df['id'] == df['id'][i])]['ai_pattern'].reset_index(drop=True)[0]

### EDA for Time Gap (days)
len(df[df['delta_dt'].isna()]) # Total 31 nan = no dates
# Remove sources longer than today. ### Need to fix this later....
today = datetime.now()
today
for i in range(len(df)):
    if df['datetime'][i] > today:
        print('id, sourceid, and datetime: ', df['id'][i],df['sourceid'][i],df['datetime'][i],df['url'][i])
        df['delta_dt'][i] = np.nan
len(df[df['delta_dt'].isna()]) # became 38: There are 7 cases which shows wrong date for source. We removed those.
# df[df['id']==39]['url']
# df[(df['id']==1164)&(df['sourceid']==3)]['delta_dt']

len(df[df['sourceid']>0]) # Total 1068 sources
len(df[(df['sourceid']>0)&(df['delta_dt'].isna())]) #So we removed 38 sources which have no or wrong dates.
df[df['sourceid']>0]['delta_dt'].value_counts().sum() # There are 1030 sources which contains delta_dt
df[df['sourceid']>0]['delta_dt'].value_counts() # sources distribution
# Time gap density distribution

df[df['sourceid']>0]['delta_dt'].value_counts().plot(kind='density')
plt.xlabel('Time Gap (days) between a fact check and its source')
plt.ylabel('Density based on 1,030 sources')
# plt.xlim([-,300])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/days_dist.png',dpi=600)
plt.show()

df[df['sourceid']>0]['delta_dt'].describe()
df[df['sourceid']>0]['delta_dt'].median()
df[df['delta_dt']<0][['delta_dt','date']]

## Time gap distribution for ratings
# Median time gap for every rating
df[df['sourceid']>0].groupby('rating')['delta_dt'].median().sort_values(ascending=False) # median time gap for each id Based on the 1068 sources (excluding fcs).
df[df['sourceid']>0]['rating'].value_counts()
df[df['rating']=='Misattributed']['url']

# Make a labels of X with (# of sources)
x = []
li_rate = df[df['sourceid']>0].groupby('rating')['delta_dt'].median().sort_values(ascending=False).index.tolist()
for i in li_rate:
    x.append(i+'('+str(df[df['sourceid']>0]['rating'].value_counts()[i])+')')

y=df[df['sourceid']>0].groupby('rating')['delta_dt'].median().sort_values(ascending=False).tolist()
plt.bar(x,y)
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Ratings (number of sources)')
plt.ylabel('Median Time Gap (Days)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/time_rating.png',dpi=600)
plt.show()

# Further seperate it into real vs. fake
real_gap = df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))]['delta_dt']) # 284
fake_gap = df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))]['delta_dt']) # 376

x = ['REAL(284)','FAKE(376)']
y = [real_gap,fake_gap]
plt.bar(x,y,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,250])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap.png',dpi=600)
plt.show()
plt.close()
# Seperate by ai vs. non ai.


r_gap_ai = df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))&(df['src_ai_pattern']==True)]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))&(df['src_ai_pattern']==True)]['delta_dt']) #242
f_gap_ai = df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))&(df['src_ai_pattern']==True)]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))&(df['src_ai_pattern']==True)]['delta_dt']) # 236

r_gap_nai = df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))&(df['src_ai_pattern'].isna())]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['TRUE','Mostly True']))&(df['src_ai_pattern'].isna())]['delta_dt']) #42
f_gap_nai = df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))&(df['src_ai_pattern'].isna())]['delta_dt'].median()
len(df[(df['sourceid']>0)&(df['rating'].isin(['FALSE','Mostly False']))&(df['src_ai_pattern'].isna())]['delta_dt']) #140

x = ['REAL(242)','FAKE(236)']
yai = [r_gap_ai,f_gap_ai]
plt.bar(x,yai,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],yai[i],yai[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,250])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap_ai.png',dpi=600)
plt.show()

x = ['REAL(42)','FAKE(140)']
ynai = [r_gap_nai,f_gap_nai]
plt.bar(x,ynai,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],ynai[i],ynai[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,250])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap_nai.png',dpi=600)
plt.show()

real_gap


fake_gap
f_gap_ai - r_gap_ai
f_gap_nai - r_gap_nai

'''
## Time gap distribution for ratings
## Mean time gap
# timegapdf = df[df['sourceid']>0].groupby('id')['delta_dt'].mean() # mean time gap for each id Based on the 1068 sources (excluding fcs).
# df['mean_delta'] = np.nan
# for i in range(len(timegapdf)):
#     df['mean_delta'][i] = timegapdf.iloc[i]
# df[df['sourceid']==0].groupby('legitimacy')['mean_delta'].mean()
# df['mean_delta']

# Median time gap for every rating
timegapdf = df[df['sourceid']>0].groupby('id')['delta_dt'].median() # median time gap for each id Based on the 1068 sources (excluding fcs).
df['med_delta'] = np.nan
for i in range(len(timegapdf)):
    df['med_delta'][i] = timegapdf.iloc[i]

df[df['sourceid']==0].groupby('legitimacy')['med_delta'].median().sort_values(ascending=False)
df[df['sourceid']==0].groupby('legitimacy')['med_delta'].median().sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/time_rating.png',dpi=600)
plt.show()

# Further seperate it into real vs. fake
fake_gap = df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))]['med_delta']) # 80
real_gap = df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))]['med_delta']) #49

x = ['REAL(49)','FAKE(80)']
y = [real_gap,fake_gap]
plt.bar(x,y,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,300])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap.png',dpi=600)
plt.show()

# Seperate by ai vs. non ai.

f_gap_ai = df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))&(df['ai_pattern']==True)]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))&(df['ai_pattern']==True)]['med_delta']) # 41
r_gap_ai = df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))&(df['ai_pattern']==True)]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))&(df['ai_pattern']==True)]['med_delta']) #36
f_gap_nai = df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))&(df['ai_pattern'].isna())]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))&(df['ai_pattern'].isna())]['med_delta']) # 39
r_gap_nai = df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))&(df['ai_pattern'].isna())]['med_delta'].median()
len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))&(df['ai_pattern'].isna())]['med_delta']) #13

x = ['REAL(36)','FAKE(41)']
yai = [r_gap_ai,f_gap_ai]
plt.bar(x,yai,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],yai[i],yai[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,300])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap_ai.png',dpi=600)
plt.show()

x = ['REAL(13)','FAKE(39)']
ynai = [r_gap_nai,f_gap_nai]
plt.bar(x,ynai,color=['green','red'])
for i in range(len(x)):
    plt.text(x[i],ynai[i],ynai[i],ha='center')
plt.xlabel('Ratings')
plt.ylabel('Median Time Gap (Days)')
plt.ylim([0,300])
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rf_medgap_nai.png',dpi=600)
plt.show()
'''
import csv
df['delta_dt']

output_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/delta_dt.csv"
header = ['delta_dt']
with open(output_csv,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for i in range(len(df['delta_dt'])):
        c.writerow([df['delta_dt'][i]])