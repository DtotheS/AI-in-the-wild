import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
import seaborn as sns

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/logically_090622_v2.csv")
# df = df.iloc[:,1:]
plt.style.use('seaborn-bright')

# y: # FCS, x: Dates (for each month)
# Finding: peak (8) reaches early compared to other websites (11).
df.groupby(['fc_year','fc_month']).count()['link']
dname = []
dcnt = []
for yy in range(2019,2023):
    for mm in range(1,13):
        dname.append("{}-{}".format(yy,mm))
        print("{}-{}".format(yy,mm))
        try:
            dcnt.append(df.groupby(['fc_year', 'fc_month']).count()['link'][yy][mm])
        except:
            dcnt.append(0)
len(dname) == len(dcnt)

plt.plot(dcnt)
plt.grid()
plt.xticks(np.arange(len(dcnt)), dname, rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("Logically: # FCs for each month (2019-6 ~ 2022-08")
plt.xlabel("Date (yyyy-mm)")
plt.ylabel("number of FCs "+"(Total # FCs: %s)" %len(df))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/fcs_month.png",dpi=600)
plt.show()
plt.close()

# y: # Fcs, x: Country
x = df.groupby(['location']).count()['link'].index.to_list()
x.append("None") # No country label
y = df.groupby(['location']).count()['link'].to_list()
y.append(df['location'].isnull().sum())
yp = []
for i in y:
    yp.append(str(round(i/len(df) * 100,2))+"%")


plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("# FCs by Country")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Country")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/fcs_country.png",dpi=600)
plt.show()
plt.close()


# y: FCs, x: ratings
x = df.groupby(['rating']).count()['link'].index.to_list()
y = df.groupby(['rating']).count()['link'].to_list()
yp = []
for i in y:
    yp.append(str(round(i/len(df) * 100,2))+"%")

plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("# FCs by Rating")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Ratings")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/fcs_rating.png",dpi=600)
plt.show()
plt.close()

# y: FCs, x: Authors
x = df.groupby('author').count().index.to_list()
y = df.groupby('author').count()['link'].to_list()
for a, b in zip(x,y):
    print(a,b)

'''Combine authors with same first name if the last name is not clearly different
Ankita Kulkarni 240
Ankita.K@Logically.Co.Uk 2

Arron Williams 3
Arron.W 2

Gayathri 92
Gayathri Loka 10

Pallavi 72
Pallavi Sethi 7

Sam Doak 4
Sam.D 5

Shreyashi Roy 9
Shreyashi.R 1
'''

namex = [("Ankita Kulkarni","Ankita.K@Logically.Co.Uk"),("Arron Williams","Arron.W"),("Gayathri Loka","Gayathri"),("Pallavi Sethi","Pallavi"),("Sam Doak","Sam.D"),("Shreyashi Roy","Shreyashi.R")]
for i in range(len(df)):
    for nn in range(len(namex)):
        if df['author'][i] == namex[nn][1]: # replace all name variations
            df['author'][i] = namex[nn][0]
# set(df['author'])

x = df.groupby('author').count().index.to_list()
y = df.groupby('author').count()['link'].to_list()

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

# FCS per author - only top 20 authors
plt.bar(x,y,label="# FCs")
for i in range(len(x)):
    plt.text(x[i], y[i], yp[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.90)
plt.title("# FCs by author - Top 20 only")
plt.ylabel("number of FCs "+"(Total # FCs: %s)"%len(df))
plt.xlabel("Author")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/fcs_20author.png",dpi=600)
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
plt.title("# authors by year")
plt.ylabel("number of authors "+"(Total: %s)"%len(set(df['author'])))
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/authors_year.png",dpi=600)
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
plt.title("AVG. # authors by year")
plt.ylabel("AVG. number of authors "+"(Total: %s)"%len(set(df['author'])))
plt.xlabel("year")
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/avgAuthors_year.png",dpi=600)
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

ratings_li = ['TRUE','PARTLY TRUE','MISLEADING','FALSE','UNVERIFIABLE']
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
plt.title("Absolute number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Rating")
plt.ylabel("# FCs")
# Show the graph
plt.legend()
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/absFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

# PercentageL top 10 authors FCs by rating
## finding: different with politifact, this graph shows that there is not much difference between authors.
num = 0
for name in x[:10]:
    num+=1
    plt.plot(ratings_li,dic_rat[name+"_p"],marker='',color=palette(num),linewidth=1, alpha=0.9, label=name)

# Add titles
plt.title("Relative(%) number of FCs by rating & author", loc='left', fontsize=12, fontweight=0, color='red')
plt.xlabel("Ratings")
plt.ylabel("Percentage (%)")
# Show the graph
plt.legend()
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/relFcs_rating_author.png",dpi=600)
plt.show()
plt.close()

## key words analysis

df['key_word'] = df['key_word'].apply(lambda x: x.strip("'[]`").split("\\n")) #make list as list.
df['key_word'] = df['key_word'].apply(lambda x : None if (x == ['']) else x) #[''] = None
len(df) - df['key_word'].isnull().sum() #2002 has keywods.

ks = []
for i in range(len(df)):
    try:
        ks.extend(df['key_word'][i])
    except:
        pass

ks = [key.lower() for key in ks]
len(ks) # 4438 keywords
len(set(ks)) # 575 different keys

import collections
counter = collections.Counter(ks)
sort_counter = dict(sorted(counter.items(), key=lambda item: item[1],reverse=True))

x = list(sort_counter.keys())[:20]
y = list(sort_counter.values())[:20]

plt.bar(x,y,label="Key Words Frequency - Top 20")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.55, top=0.90)
plt.title("Key Words")
plt.ylabel("Frequency"+"(Total frequency: %s)"%len(ks))
plt.xlabel("Key Word (Total # keywords: %s)" %len(set(ks)))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/keyword_freq.png",dpi=600)
plt.show()
plt.close()

# Sources Analysis
import ast
df['src_name'] = df['src_name'].apply(lambda x: ast.literal_eval(x))
# df['src_name'] = df['src_name'].apply(lambda x: x.strip("'[]`").split("\\n")) #make list as list.
df['src_name'] = df['src_name'].apply(lambda x : None if (x == ['']) else x) #[''] = None
# df[df['src_name'].isnull()]['link'].iloc[0] # no source link: only one FC does not have source.

ks = []
for i in range(len(df)):
    try:
        ks.extend(df['src_name'][i])
    except:
        pass

ks = [key.lower() for key in ks]
len(ks) # sources frequencies: 15520
len(set(ks)) # unique sources: 3677

import collections
counter = collections.Counter(ks)
sort_counter = dict(sorted(counter.items(), key=lambda item: item[1],reverse=True))

x = list(sort_counter.keys())[:20]
y = list(sort_counter.values())[:20]
x[5] = "cdc" # too long
x[19] = "u.s. fda" # too long
plt.bar(x,y,label="Sources & References - Top 20")
for i in range(len(x)):
    plt.text(x[i], y[i], y[i], ha = 'center')
plt.xticks(np.arange(len(x)),x,rotation=90)
plt.subplots_adjust(bottom=0.5, top=0.90)
plt.title("Sources & References - Top 20")
plt.ylabel("Frequency"+"(Total frequency: %s)"%len(ks))
plt.xlabel("Source Name (Total # keywords: %s)" %len(set(ks)))
plt.savefig("/Users/agathos/DtotheS/AI-in-the-wild/img/logically/source_freq.png",dpi=600)
plt.show()
plt.close()

