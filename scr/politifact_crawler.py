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

driver = webdriver.Chrome('chromedriver', options=options)
# driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
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
finaldf.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v4_092722.csv", index=False)

######################## Select only 2016 ~ 2021 years' data #######################
import os
os.getcwd()
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v4_092722.csv")
len(df) # Total 21595
# df = df.rename(columns={df.columns[0]: "id" })
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

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v4_092722.csv", index=False)
# len(df)
# df = df[df['fc_year'].between(2016,2021)] # Select FCs published between 2016 and 2021
# df = df[df['fc_year']==2022] # Select 2022
# len(df) # total # FCs: 9534
# df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv3_16to21.csv",index=False)
# df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pf2022.csv",index=False)

##################### Crawl FC article body contents ##########################
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv4_16to21.csv")
# df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pf2022.csv")
# df = df[df['rating'].isin(['barely-true', 'false','pants-fire'])]
# df = df.reset_index(drop=True)

df[['title','tags','summary','bodyt','sources_num','sources']] = None

for i in range(len(df)):
    # driver.implicitly_wait(10)
    url = df['link'][i]
    driver.get(url)

    try:
        title = driver.find_element_by_class_name("c-title").text
        df['title'][i] = title
    except:
        pass

    try:
        tags = driver.find_elements_by_class_name("m-list__item")  # 4
        tags = [x.text.strip() for x in tags]
        df['tags'][i] = tags
    except:
        pass

    try:
        summary = driver.find_element_by_class_name("short-on-time").text
        df['summary'][i] = summary
    except:
        pass

    try:
        bodyt = driver.find_element_by_class_name("m-textblock").text
        df['bodyt'][i] = bodyt
    except:
        pass

    try:
        # sources = driver.find_element_by_class_name("m-superbox__content").text.split("\n")
        sources = driver.find_element_by_class_name("m-superbox__content").text
        sources = list(filter(bool, sources.strip().splitlines())) # this additional step will solve "\n\n" issue.
        sources = [x.strip() for x in sources]
        df['sources'][i] = sources
    except:
        pass

    try:
        snum = len(sources)
        df['sources_num'][i] = snum
    except:
        pass
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/pfv5_16to21.csv",index=False)

'''
tag_li = df['tags'].tolist()
tags = [x.strip() for clist in tag_li for x in clist]
tag_li = set(tags)

"GUNS" in tag_li
"ECONOMY" in tag_li
"ABORTION" in tag_li

df['contain'] = None
for i in range(len(df)):
    df['contain'][i] = any(ele in df['tags'][i] for ele in ["ABORTION","ECONOMY","GUNS"])

df[df['contain']].to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/fakenews_cbv2.csv",index=False)   
'''