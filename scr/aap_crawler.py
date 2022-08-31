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
df['id'] = range(1,len(links)+1)
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