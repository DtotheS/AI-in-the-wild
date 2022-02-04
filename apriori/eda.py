import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

src = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/fc_src.csv") # fc + sources data
fc = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/factcheck_websites.csv") # 1203 fc data
### Load df which include delta_dt (time gap days), ai_pattern (all ai), ai_pattern_ent (NE ai)
df = pd.read_pickle('/Users/agathos/DtotheS/AI-in-the-wild/apriori/df3.pkl') # contains only 198 fc and their sources
# df['rating'] match id and its rating for all sources.
df['rating'] = np.nan
for i in range(len(df)):
    df['rating'][i] = df.loc[(df['sourceid']==0)&(df['id'] == df['id'][i])]['legitimacy'].reset_index(drop=True)[0]

# assign T/F for sources if there is ai pattern for its fact check.
df['src_ai_pattern'] = np.nan
for i in range(len(df)):
    df['src_ai_pattern'][i] = df.loc[(df['sourceid']==0)&(df['id'] == df['id'][i])]['ai_pattern'].reset_index(drop=True)[0]


len(src) # Total = 2271
len(fc) #FC = 1203 => src = 2271 - 1203 = 1068
len(fc[fc['title']=="none"]['id'])
len(fc[fc['url']=="none"]['id'])
len(src[src['title']=="none"]['id']) # Used "" to collect title. Among 1068 only 99 were not collected. But, among collected, there may be dummy titles.
len(src[src['url']=="none"]['id']) #Among 1068, 718 does not contains url. That is mainly because they did not contain urls before 12/21/2019

# Making a labels of x by combining list of ratings and list of values e.g.) False (35)
def conc(name,num):
    labels = []
    li_rate = name
    for i in li_rate:
        labels.append(str(i)+'('+str(num[i])+')')
    return labels

# name = df[df['sourceid']>0].groupby('rating')['delta_dt'].median().sort_values(ascending=False).index.tolist()
# num = df[df['sourceid']>0]['rating'].value_counts()[i]

# Ratings distribution for 1203
fc['legitimacy'].value_counts() #absolute
y = fc['legitimacy'].value_counts(normalize=True).mul(100).round(decimals=1).tolist() #perc
name = fc['legitimacy'].value_counts().index.tolist()
num =  fc['legitimacy'].value_counts()
x = conc(name,num)
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Ratings of 1,203 fact checks (number of fact checks)')
plt.ylabel('Density (%, base = 1,203)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rating_dist_1203per.png',dpi=600)
plt.show()
plt.close()
# x = fc.groupby('legitimacy').count()['id'].index.to_list()
# y = fc.groupby('legitimacy').count()['id'].to_list()

## Ratings distribution for 198.
df[df['sourceid']==0]['legitimacy'].value_counts().sum() #total 198 which contains at least one source
len(df[df['sourceid']==0]['legitimacy'])
name = df[df['sourceid']==0]['legitimacy'].value_counts().index.tolist()
num = df[df['sourceid']==0]['legitimacy'].value_counts()
y=df[df['sourceid']==0]['legitimacy'].value_counts(normalize=True).mul(100).round(decimals=1).tolist()
x = conc(name,num)
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Ratings of 198 fact checks which contain at least on source (number of fact checks)')
plt.ylabel('Density (%, base = 198)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/rating_dist_198per.png',dpi=600)
plt.show()
plt.close()


## Distribution of the number of sources for 198 fact checks
name = fc[fc['sources_num']>0]['sources_num'].value_counts().sort_index().index.tolist()
num = fc[fc['sources_num']>0]['sources_num'].value_counts().sort_index()
y = fc[fc['sources_num']>0]['sources_num'].value_counts(normalize=True).mul(100).sort_index().round(decimals=1).tolist()
x = conc(name,num)
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Number of sources for each fact check (count of fact checks)')
plt.ylim([0,22])
plt.ylabel('Density (%, base = 198)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/numsources_dist.png',dpi=600)
plt.show()
plt.close()

## Real: Distribution of the number of sources for 198 fact checks
name = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['TRUE', 'Mostly True']))]['sources_num'].value_counts().sort_index().index.tolist()
num = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['TRUE', 'Mostly True']))]['sources_num'].value_counts().sort_index()
y = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['TRUE', 'Mostly True']))]['sources_num'].value_counts(normalize=True).mul(100).sort_index().round(decimals=1).tolist()
x = conc(name,num)
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Number of sources for each REAL fact check (count of REAL fact checks)')
plt.ylim([0,22])
plt.ylabel('Density (%, base = 49)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/numsources_dist_real.png',dpi=600)
plt.show()
plt.close()

## Fake: Distribution of the number of sources for 198 fact checks
name = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['FALSE', 'Mostly False']))]['sources_num'].value_counts().sort_index().index.tolist()
num = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['FALSE', 'Mostly False']))]['sources_num'].value_counts().sort_index()
y = fc[(fc['sources_num']>0)&(fc['legitimacy'].isin(['FALSE', 'Mostly False']))]['sources_num'].value_counts(normalize=True).mul(100).sort_index().round(decimals=1).tolist()
x = conc(name,num)
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(rotation='vertical')
plt.xlabel('Number of sources for each FAKE fact check (count of FAKE fact checks)')
plt.ylim([0,22])
plt.ylabel('Density (%, base = 80)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/numsources_dist_fake.png',dpi=600)
plt.show()
plt.close()

## Number of sources distribution grouped by legitimacy (based on 198 fact checks legitimacy for 1,068 sources)
y = df[df['sourceid']==0][['legitimacy','sources_num']].astype({'sources_num':'int'}).groupby('legitimacy').sum().sort_values('sources_num',ascending=False)['sources_num'].tolist()# total source numbers
name = df[df['sourceid']==0][['legitimacy','sources_num']].astype({'sources_num':'int'}).groupby('legitimacy').sum().sort_values('sources_num',ascending=False).index.tolist()
num = df[df['sourceid']==0]['legitimacy'].value_counts() # fact checks number (total 198)
x = conc(name,num)
# y2_sort = []
# for r in name:
#     y2_sort.append(y2.loc[r][0])
#     print(r,y2.loc[r][0])
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(x, rotation='vertical')
plt.xlabel('Ratings of 198 fact checks (number of fact checks)')
plt.ylabel('Number of sources')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/num_srcs.png.png',dpi=600)
plt.show()
plt.close()

## Average number of sources distribution grouped by legitimacy (based on 198 fact checks legitimacy for 1,068 sources)
y = df[df['sourceid']==0][['legitimacy','sources_num']].astype({'sources_num':'int'}).groupby('legitimacy').mean().round(decimals=1).sort_values('sources_num',ascending=False)['sources_num'].tolist() # mean source numbers
name = df[df['sourceid']==0][['legitimacy','sources_num']].astype({'sources_num':'int'}).groupby('legitimacy').mean().sort_values('sources_num',ascending=False).index.tolist()
num = df[df['sourceid']==0]['legitimacy'].value_counts() # fact checks number (total 198)
x = conc(name,num)
# y2_sort = []
# for r in name:
#     y2_sort.append(y2.loc[r][0])
#     print(r,y2.loc[r][0])
plt.bar(x,y, color='#1f77b4')
for i in range(len(x)):
    plt.text(x[i],y[i],y[i],ha='center')
plt.xticks(x, rotation='vertical')
plt.xlabel('Ratings of 198 fact checks (number of fact checks)')
plt.ylabel('Average number of sources \n(# sources / # fact checks)')
plt.tight_layout()
plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/mean_num_srcs.png',dpi=600)
plt.show()
plt.close()





'''
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
'''

df.columns
## EDA
# Among all ai: 56.1%
df['ai_pattern'].sum()
df[df['sourceid']==0]['ai_pattern'].sum() / len(df[df['sourceid']==0])# among 198, 111 have ai pattern.

# Among NE to Claim ai: 33.8%
df['ai_pattern_ent_claim'].sum()
df[df['sourceid']==0]['ai_pattern_ent_claim'].sum() / len(df[df['sourceid']==0])# among 198, only 67 have ne ai pattern.

## Find percentage of existence of ai pattern for each label.
# All AI pattern
lbls = list(set(df['legitimacy'].tolist()))
for i in lbls:
    print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern']))*100) + "%)")

# NE AI Pattern
lbls = list(set(df['legitimacy'].tolist()))
for i in lbls:
    print(str(i)+": " + str(len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent_claim'])) + " vs." + str(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent_claim'].count()) + " ("+ str((df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent_claim'].count()/len(df[(df['sourceid']==0)&(df['legitimacy']==i)]['ai_pattern_ent_claim']))*100) + "%)")


### EDA for Time Gap (days)
'''
### df2 pikle already done.
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
'''

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

### Common B words

all_b = []
claim_b = []
for i in range(len(df[df['sourceid']==0])):
    all_b.extend(df['ai_Bwords'][i])
    claim_b.extend(df['ai_netk_claim_Bwords'][i])

all_b_fake = []
claim_b_fake = []
for i in range(len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))])):
    all_b_fake.extend(df['ai_Bwords'][i])
    claim_b_fake.extend(df['ai_netk_claim_Bwords'][i])

all_b_real = []
claim_b_real = []
for i in range(len(df[(df['sourceid']==0)&(df['legitimacy'].isin(['TRUE','Mostly True']))])):
    all_b_real.extend(df['ai_Bwords'][i])
    claim_b_real.extend(df['ai_netk_claim_Bwords'][i])
# df[(df['sourceid']==0)&(df['legitimacy'].isin(['FALSE','Mostly False']))]

# wlist = all_patterns4_words['id2']
def fig_b(word_list,file_name):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    # convert list to string and generate
    unique_string = (" ").join(word_list)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white', min_font_size=15, collocations=False).generate(unique_string)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('/Users/agathos/DtotheS/AI-in-the-wild/apriori/img/%s.png' % (file_name), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

fig_b(claim_b_real,'bs_real_claim')

'''
all_patterns4_words.items()
# Make a CSV file and List of AB & BC patterns.
output_csv4 = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/words.csv"
all_patterns4_words.items()
header = ['id','keywords B']
with open(output_csv4,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    for key, value in all_patterns4_words.items():
        c.writerow([key,value])
'''

from collections import Counter
# c = Counter(all_b)
# c.most_common(10)[2][1]
# len(all_b)
var_li = [all_b,claim_b,all_b_real,claim_b_real,all_b_fake,claim_b_fake,]
for i in var_li:
    print(len(i))
    c=Counter(i)
    print(c.most_common(10))
    print([round(c[j[0]] / len(i) *100,1) for j in c.most_common(10)])

df.columns
df[df['ai_pattern_ent_claim']==True].groupby('legitimacy').count()