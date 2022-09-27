import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
import seaborn as sns

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
df.columns
len(df) #843

df['fc_date'] = pd.to_datetime(df['fc_date'])
df = df[df['fc_date'].between(dt(2019,5,17),dt(2022,8,31))] # 827

df.isnull().sum() # sum links are missing summarized claims.

plt.style.use('seaborn-bright')

# y: # FCS, x: Dates (for each month)
# Finding: peak (8) reaches early compared to other websites (11).
df.groupby(['fc_year','fc_month']).count()['link']
dname = []
dcnt = []
for yy in range(2018,2023):
    for mm in range(1,13):
        dname.append("{}-{}".format(yy,mm))
        print("{}-{}".format(yy,mm))
        try:
            dcnt.append(df.groupby(['fc_year', 'fc_month']).count()['link'][yy][mm])
        except:
            dcnt.append(0)
len(dname) == len(dcnt)

dname = dname[16:-4]
dcnt = dcnt[16:-4] # 2018-12 ~ 2022-08

plt.plot(dcnt,linestyle="-", marker="o")
plt.grid()
plt.xticks(np.arange(0,len(dcnt),5), dname[::5], rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("AAP: # FCs for each month (2019-5 ~ 2022-08)")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("number of FCs "+"(Total # FCs: %s)" %len(df))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_fcs_month.png",dpi=600)
plt.show()
plt.close()

# y: FCs, x: ratings
x = df.groupby(['rating']).count()['link'].index.to_list()
y = df.groupby(['rating']).count()['link'].to_list()
yp = []
for i in y:
    yp.append(str(int(round(i/len(df) * 100,0)))+"%")

plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("AAP: # FCs by Rating")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Ratings")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_fcs_rating.png",dpi=600)
plt.show()
plt.close()

# y: FCs, x: Authors
x = df.groupby('author').count().index.to_list()
y = df.groupby('author').count()['link'].to_list()

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
    yp.append(str(int(round(i/len(df) * 100,0)))+"%")

# FCS for each author - only top 20 authors
plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.5, top=0.90)
plt.title("AAP: # FCs by author - Top 20 only")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Author")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_fcs_20author.png",dpi=600)
plt.show()
plt.close()

# authors for each year
years_li = df.groupby('fc_year').count().index.to_list()
y = []
for yy in years_li:
    print(str(yy) +": # authors = "+str(len(df.groupby(['fc_year','author']).count().sort_values(['fc_year','link'],ascending=False).loc[yy])))
    y.append(len(df.groupby(['fc_year','author']).count().sort_values(['fc_year','link'],ascending=False).loc[yy]))
x = [str(yy) for yy in years_li]

plt.bar(x,y,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("AAP: # authors by year")
plt.ylabel("number of authors "+"(Total: %s)"%len(set(df['author'])))
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_authors_year.png",dpi=600)
plt.show()
plt.close()

# average fc/author for each year
y2 = df.groupby('fc_year').count()['link'].to_list() # number of FCs per year
for i in range(len(y)): # here, y is defined by # author per year. from the above code.
    print(str(x[i])+"'s FCs per Author: "+str(round(y2[i]/y[i],2)))
    y2[i] = round(y2[i]/y[i],2)

plt.bar(x,y2,label="# Authors")
for i in range(len(x)):
    plt.text(x[i], y2[i], y2[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.90)
plt.title("AAP: FCs/Authors by year")
plt.ylabel("Number of FCs per author")
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_fcs_perauthor.png",dpi=600)
plt.show()
plt.close()

# Ratings of top20 authors
y = df.groupby(['author']).count().sort_values(['link'],ascending=False)['link'][:20].to_list()
x = df.groupby(['author']).count().sort_values(['link'],ascending=False)[:20].index.tolist()

df_aurating = pd.DataFrame()
df_aurating['dummy'] = range(len(set(df['rating'])))
for nn in x:
    df_aurating['%s' % nn] = pd.Series(df.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].index.get_level_values('rating'))
    df_aurating['%s_v' % nn] = pd.Series(df.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array)  # since this is Series, .array is needed to add as column in df.
    df_aurating['%s_p' % nn] = pd.Series((df.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array / df.groupby(['author','rating']).count().sort_values(['author','link'],ascending=False).loc[nn]['link'][:].array.sum()) * 100)

ratings_li = list(set(df['rating']))
ratings_li = ['true','mostly true','mixture','mostly false','false',
              'partly false','somewhat true','somewhat false','misleading',
              'altered image', 'missing context', 'altered photo/video', 'satire', 'altered photo', 'ambiguous']

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

## top 10 authors FCs by rating
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("AAP: Absolute number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Rating")
plt.ylabel("# FCs")
# Show the graph
plt.legend()
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.35, top=0.90)
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_absFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

# PercentageL top 10 authors FCs by rating
## finding: different with politifact, this graph shows that there is not much difference between authors.
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name+"_p"],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("AAP: Relative(%) number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Ratings")
plt.ylabel("Percentage (%)")
# Show the graph
plt.legend()
plt.xticks(np.arange(len(ratings_li)),ratings_li,rotation=90)
plt.subplots_adjust(bottom=0.35, top=0.90)
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/aap/s_relFcs_rating_author.png",dpi=600)
plt.show()
plt.close()