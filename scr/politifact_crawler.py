## Set up
# !brew install chromedriver
# !conda install selenium
# !pip install beautifulsoup4

## Modules
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import csv
import time
from os.path import exists
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
import re

# driver = webdriver.Chrome(ChromeDriverManager().install())
# driver.close()

# Define Functions

# Start to Collect Data
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

# driver = webdriver.Chrome('chromedriver', options=options)
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
# driver.get("https://www.politifact.com/factchecks/list/?page=2&ruling=true")
# driver.current_url

ratings = ['true','mostly-true','half-true','barely-true','false','pants-fire']
# ['false']
li_rows = []

for rating in ratings:
    print(rating)
    driver.get("https://www.politifact.com/factchecks/list/?page=1" + "&ruling=" + rating)
    print(driver.current_url)
    wait = WebDriverWait(driver, 10)
    result = driver.find_element_by_class_name("o-listicle__list")
    for items in result.find_elements_by_class_name('o-listicle__item'):
        where = items.find_elements_by_class_name('m-statement__name')[0].text
        when = items.find_elements_by_class_name('m-statement__desc')[0].text
        title = items.find_elements_by_class_name('m-statement__quote')[0].text
        link = items.find_elements_by_class_name('m-statement__quote')[0].find_elements_by_tag_name('a')[0].get_attribute("href")
        author = items.find_elements_by_class_name('m-statement__footer')[0].text
        list_cells = [where, when, title, link, rating, author]
        li_rows.append(list_cells)

    nextpage = driver.find_elements_by_class_name("m-list__item")[0] # for page=1, the first button is "Next" button
    while True:
        nextpage.click()
        print(driver.current_url)
        wait = WebDriverWait(driver, 10)

        try:
            result = driver.find_element_by_class_name("o-listicle__list")
            for items in result.find_elements_by_class_name('o-listicle__item'):
                where = items.find_elements_by_class_name('m-statement__name')[0].text
                when = items.find_elements_by_class_name('m-statement__desc')[0].text
                title = items.find_elements_by_class_name('m-statement__quote')[0].text
                link = items.find_elements_by_class_name('m-statement__quote')[0].find_elements_by_tag_name('a')[
                    0].get_attribute("href")
                author = items.find_elements_by_class_name('m-statement__footer')[0].text
                list_cells = [where, when, title, link, rating, author]
                li_rows.append(list_cells)
            nextpage = driver.find_elements_by_class_name("m-list__item")[
                1]  # from page=2, the second button is "Next" button
        except:
            try:
                nextpage = driver.find_elements_by_class_name("m-list__item")[1] #from page=2, the second button is "Next" button
            except:
                break

driver.quit()
li_rows

''' When we know the last page number
for rating in ratings:
    for pn in range(1, 190):
        print("Rating: " + rating + ", Page: " + str(pn))
        # print("https://www.politifact.com/factchecks/list/?page=" + str(pn) + "&ruling=" + rating)

        # wait = WebDriverWait(driver, 10)
        driver.get("https://www.politifact.com/factchecks/list/?page=" + str(pn) + "&ruling=" + rating) # get start~end pages for each rating
        result = driver.find_element_by_class_name("o-listicle__list")

        for items in result.find_elements_by_class_name('o-listicle__item'):
            where = items.find_elements_by_class_name('m-statement__name')[0].text
            when = items.find_elements_by_class_name('m-statement__desc')[0].text
            title = items.find_elements_by_class_name('m-statement__quote')[0].text
            link = items.find_elements_by_class_name('m-statement__quote')[0].find_elements_by_tag_name('a')[0].get_attribute("href")
            author = items.find_elements_by_class_name('m-statement__footer')[0].text
            list_cells = [where, when, title, link, rating, author]

            li_rows.append(list_cells)
'''

df = pd.DataFrame(li_rows, columns=['whoc', 'when', 'claim', 'link', 'rating', 'whofc'])
# foo = df['when'][234].replace(".","").replace(":","")
# foo.split("on")

claim_date = []
claim_where = []
for i in range(len(df)):
    foo = df['when'][i].replace(".","").replace(":","")
    foo = foo.split("on",1)[1].split("in",1)
    try:
        claim_date.append(foo[0].strip())
    except:
        claim_date.append(None)

    try:
        claim_where.append(foo[1].strip())
    except:
        claim_where.append(None)

df['cdate'] = claim_date
df['cwhere'] = claim_where


fc_author = []
fc_date = []
for i in range(len(df)):
    foo1 = df['whofc'][i].split("By")[1].split("•")[0].strip()
    foo2 = df['whofc'][i].split("By")[1].split("•")[1].strip()
    try:
        fc_author.append(foo1)
    except:
        fc_author.append(None)

    try:
        fc_date.append(foo2)
    except:
        fc_date.append(None)

df['author'] = fc_author
df['fc_date'] = fc_date

finaldf = df[['claim','link','rating','fc_date','author','cdate','cwhere']]

# import os
# os.getcwd()
finaldf.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v3_072122.csv")

############ EDA ############
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import csv
os.getcwd()
df = pd.read_csv("./ai-in-the-wild/data/politifact_v3_072122.csv")
len(df) # Total 17867
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
len(df) # total # FCs: 8353

years_li=list(set(df['fc_year']))
years_li = [int(x) for x in years_li]
years_li.sort()

month_li = list(set(df['fc_month']))
month_li = [int(x) for x in month_li]
month_li.sort()

## rating
rating = df.groupby(['fc_year','rating']).count().sort_values(['fc_year'],ascending=False)#.loc[2016]
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
for yy in [2016,2017,2018,2019,2021]:
    plt.plot(rating_dic['name'], rating_dic[yy], linestyle="-", marker="o", label="%s" %(yy))
plt.legend()
plt.xticks(np.arange(len(rating_dic['name'])), rating_dic['name'], rotation=90)
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.show()
plt.close()

# number of FCs: years comparison x: months
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
plt.show()

# number of FCs: x: years
cnt_years=[]
for yy in years_li:
    cnt_years.append(count.loc[yy].sum()[0])
plt.bar(years_li, cnt_years,label="# FCs")
# plt.xticks(np.arange(len(years_li)),years_li,color='blue')
# plt.subplots_adjust(bottom=0.2, top=0.99)
plt.legend()
plt.show()
plt.close()

# rating: Aggregate the years
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