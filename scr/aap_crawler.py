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

# Start to Collect Data
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

driver = webdriver.Chrome('chromedriver', options=options)
# driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
driver.get("https://www.aap.com.au/factcheck/page/43/")
# driver.current_url
# driver.close()

'''
# For 1st page
urls = driver.find_elements_by_class_name("a-content")
for i in range(len(urls)):
    print(urls[i].find_elements_by_tag_name('a')[0].get_attribute("href"))

# 2nd page ~
urls = driver.find_elements_by_class_name("m-t-0")
for i in range(len(urls)):
    print(urls[i].find_elements_by_tag_name('a')[0].get_attribute("href"))
'''

# urls crawling for all pages (total 43 pages)
links = [] # 41 * 20 + 16 (last page) + 5 (first page) = 841
for i in range(43):
    driver.get("https://www.aap.com.au/factcheck/page/%s/" %(i+1))
    if i == 0: # for the 1st page, it has different page layout.
        urls = driver.find_elements_by_class_name("a-content")
        for ele in range(len(urls)):
            links.append(urls[ele].find_elements_by_tag_name('a')[0].get_attribute("href"))
    else: # 2nd page ~
        urls = driver.find_elements_by_class_name("m-t-0")
        for ele in range(len(urls)):
            links.append(urls[ele].find_elements_by_tag_name('a')[0].get_attribute("href"))

# id,title,url,claim,rating,content_owner,author_name,date_published,date_updated,primary_category,tags,sourceid,sources_num,sources,bodyt,page_type,yearp,monthp,dayp
df = pd.DataFrame()
df['link'] = links
df['site'] = "AAP"

titles = []
authors = []
claims = []
ratings = []
fc_dates = []
for i in range(len(df)):
    driver.get(df['link'][i])
    titles.append(driver.find_element_by_tag_name("h1").text)  # title
    authors.append(driver.find_element_by_class_name("info").find_elements_by_tag_name("span")[0].text)  # author name
    fc_dates.append(driver.find_element_by_class_name("info").find_elements_by_tag_name("span")[1].text)  # FC date
    try:
        claims.append(driver.find_elements_by_class_name("inner")[0].text.strip().split("\n")[1])  # claim
    except:
        claims.append(None)     # page15 ~ : no claim, but only statement.
    try:
        ratings.append(driver.find_element_by_class_name("c-article__verdict").find_element_by_tag_name("strong").text)  # rating
    except:
        ratings.append(None)
    # driver.find_elements_by_class_name("inner")[1].text.strip().split("\n")[1].strip().split(". ")[1] # Rating Reason summary
    print(str(i) +" "+df['link'][i] +" done")

if len(df) == len(titles) == len(authors) == len(claims) == len(ratings) == len(fc_dates):
    print("ok")
else:
    print("len diff")

sum(x is None for x in ratings) #4 None values - errors

for i in range(len(ratings)):
    if ratings[i] == None:
        print(str(i) + " "+ links[i])

# Mannually added
ratings[269] = "False"
ratings[551] = "Somewhat True"
ratings[566] = "False"
ratings[649] = "False"

df['title'] = titles
df['claim'] = claims
df['rating'] = ratings
df['author'] = authors
df['fc_date'] = fc_dates

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_083022.csv", index=False)

##########################Insert updated links at the top#################################################
'''
 Condition 1: Do not know the total number of links before finishing to collect data.
    C2: Need to assign largest negative number to the latest link
    C3: Crawling starts from the latest link.
    C4: The oldest link should be indexed as -1. 
 
 Solution:
    First, assign the latest: 0 1 2 3 4 5 6 .... : the oldest (let's say total 7 links)
    0-7 = -7
    1-7 = -6
    ...
    6-7 = -1
'''

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_083022.csv")
del df['id']
df.columns

links = ["https://www.aap.com.au/factcheck/brain-damage-link-to-msg-a-salty-dose-of-misinformation/","https://www.aap.com.au/factcheck/microwave-meme-cooks-up-cancer-food-fears/"] # Same older with existing crawlers: latest to oldest

# Can use excatly same crwaling method
df2 = pd.DataFrame()
df2['link'] = links
df2['site'] = "AAP"

titles = []
authors = []
claims = []
ratings = []
fc_dates = []
for i in range(len(df2)):
    driver.get(df2['link'][i])
    titles.append(driver.find_element_by_tag_name("h1").text)  # title
    authors.append(driver.find_element_by_class_name("info").find_elements_by_tag_name("span")[0].text)  # author name
    fc_dates.append(driver.find_element_by_class_name("info").find_elements_by_tag_name("span")[1].text)  # FC date
    try:
        claims.append(driver.find_elements_by_class_name("inner")[0].text.strip().split("\n")[1])  # claim
    except:
        claims.append(None)     # page15 ~ : no claim, but only statement.
    try:
        ratings.append(driver.find_element_by_class_name("c-article__verdict").find_element_by_tag_name("strong").text)  # rating
    except:
        ratings.append(None)
    # driver.find_elements_by_class_name("inner")[1].text.strip().split("\n")[1].strip().split(". ")[1] # Rating Reason summary
    print(str(i) +" "+df2['link'][i] +" done")

df2['title'] = titles
df2['claim'] = claims
df2['rating'] = ratings
df2['author'] = authors
df2['fc_date'] = fc_dates

ii = len(df2)
for i in range(len(df2)):
    df.loc[i-ii] = df2.loc[i]

df.index = df.index + len(df2)
df = df.sort_index()

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv", index=False)

################ Rating Update ###################

df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
rcan = list(set(df['rating']))
for i in range(len(rcan)):
    print(str(i) + " " + rcan[i])
len(rcan[27]) # 19: max correct label len.

# driver.get(df[df['rating']==rcan[5]]['link'].iloc[0])
import json
import re
driver.get("https://www.aap.com.au/factcheck/do-eating-disorders-have-the-highest-mortality-rate-of-any-mental-illness/")

# Method 1. Use reviewRating in the script. However, only recent FCs contain this.
eles = driver.find_elements_by_xpath('//script[@type="application/ld+json"]')
eles[-1].get_attribute("innerHTML") # 0 gives wrong json
dic = json.loads(eles[-1].get_attribute("innerHTML"))
try:
    dic[0].keys() # only 1 element, but it stored as list.
except:
    rating_ex = dic['reviewRating']['alternateName']  # this gives rating & explanation together
else:
    rating_ex = dic[0]['reviewRating']['alternateName']  # this gives rating & explanation together

# dic[0].keys()
rat_li = re.split('–|–|-\.',rating_ex)
rat_li[0].strip()

# Method 2. Use "–" as a rating indicator
## not that useful, since COVID-19, Fact-Checking cases.

df.columns
driver.get("https://www.aap.com.au/factcheck/post-covid-19-injection-syndrome-is-only-garbled-misinformation/")
ps = driver.find_element_by_class_name('c-article__verdict').find_elements_by_tag_name("p")
# ps[2].text
for i in ps:
    sobj = re.search('[a-zA-Z].{0,100}(–|–|-)',i.text).group()
    x = re.split("–|–|-", sobj)
    label = x[0].strip()
    if len(label) < 20:
        print(label)

############################################# Updated Rating Crawling ################
import json
import re
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
ratings2 = []
for i in range(len(df)):
    driver.get(df['link'][i])
    # Method 1. Use reviewRating in the script. However, only recent FCs contain this.
    eles = driver.find_elements_by_xpath('//script[@type="application/ld+json"]')
    eles[-1].get_attribute("innerHTML")  # 0 gives wrong json
    dic = json.loads(eles[-1].get_attribute("innerHTML"))
    try: # when there are multiple dictionaries
        # dic[0].keys()  # only 1 element, but it stored as list.
        rating_ex = dic[0]['reviewRating']['alternateName']  # this gives rating & explanation together
        rat_li = re.split('–|–|-|\.', rating_ex)
        ratings2.append(rat_li[0].strip())
    except:
        try: # when there is only one dictionary
            rating_ex = dic['reviewRating']['alternateName']  # this gives rating & explanation together
            rat_li = re.split('–|–|-|\.', rating_ex)
            ratings2.append(rat_li[0].strip())
        except: # Method 2
            ps = driver.find_element_by_class_name('c-article__verdict').find_elements_by_tag_name("p")
            foo1 = 0
            # ps[2].text
            for tt in ps:
                sobj = re.search('[a-zA-Z].{0,100}(–|–|-)', tt.text)
                if sobj:
                    sobj = sobj.group()
                    x = re.split("–|–|-", sobj)
                    label = x[0].strip()
                    if len(label) < 20:
                        ratings2.append(label)
                        foo1 += 1
            if foo1 == 0: # for some cases, rating was created using <li></li>
                ps = driver.find_element_by_class_name('c-article__verdict').find_elements_by_tag_name("li")
                foo2 = 0
                # ps[2].text
                for tt in ps:
                    sobj = re.search('[a-zA-Z].{0,100}(–|–|-)', tt.text)
                    if sobj:
                        sobj = sobj.group()
                        x = re.split("–|–|-", sobj)
                        label = x[0].strip()
                        if len(label) < 20:
                            ratings2.append(label)
                            foo2 += 1
                if foo2 == 0:
                    ratings2.append(None)
    print(str(i) +" "+df['link'][i] +" done")

if len(df) == len(ratings2):
    print("ok")
else:
    print("len diff")

# lowering all char. : e.g., Partly False vs. partly false
ratings3 = []
for i in range(len(ratings2)):
    if ratings2[i]:
        ratings3.append(ratings2[i].lower())
    else:
        ratings3.append(None)

set_rat = list(set(ratings3))
for i in range(len(set_rat)):
    print(str(i)+" "+str(set_rat[i]))

ratings2.count(None) == ratings3.count(None) # double checks
wrong_rat = [1,3,7,8,9,16,19] # index numbers for the wrong ratings from set_rat.
for i in wrong_rat:
    print(ratings3.count(set_rat[i]))
# 0: None => 42 cases. All other indecies only have 1 case.

for i in wrong_rat: # to check the link and rating manually
    print(df['link'][ratings3.index(set_rat[i])])

right_rat = ['false','mostly false','false','false','false','false','partly false'] # manually collect correct ratings

rr = 0
for i in wrong_rat:
    print(ratings3[ratings3.index(set_rat[i])], " ", right_rat[rr])
    ratings3[ratings3.index(set_rat[i])] = right_rat[rr] # assign correct labels manually
    rr+=1

df['rating'] = None
df['rating'] = ratings3

import os
os.getcwd()

len(df) # Total 843
df.columns
df.isnull().sum()
df[df['author'].isnull()]

df['fc_date']=pd.to_datetime(df['fc_date'])
df['fc_year'] = df.apply(lambda row : row['fc_date'].year, axis=1)
df['fc_month'] = df.apply(lambda row : row['fc_date'].month, axis=1)
df['fc_day'] = df.apply(lambda row : row['fc_date'].day, axis=1)

df['fc_year'].isnull().sum()
df['fc_month'].isnull().sum()
df['fc_day'].isnull().sum()
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv",index=False)

'''
######################## Select only 2019 ~ 2021 years' data #######################
len(df)
df = df[df['fc_year'].between(2019,2021)] # Select FCs published between 2019 and 2021
len(df) # total # FCs: 612
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_083022_19_21.csv",index=False)
df.columns
'''

##################### author, body crawl #################
df = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv")
df[['author','body']] = None
for i in range(len(df)):
    driver.get(df['link'][i])
    try:
        at = driver.find_element_by_class_name("info").find_elements_by_tag_name("span")[0].text
        df['author'][i] = at
        print(at)
    except:
        pass

    try:
        bd = driver.find_element_by_class_name("c-article__content").text
        df['body'] = bd
        print(bd[:10])
    except:
        pass
    print(str(i)+" "+df['link'][i] + " done")


# df[df['author']==''] # 1 author is missing
# df['author'][748] = None
# df.isnull().sum()

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/aap_090122.csv",index=False)