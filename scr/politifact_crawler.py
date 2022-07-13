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

ratings = ['false']
    # ['true','mostly-true','half-true','barely-true','false','pants-fire']
li_rows = []

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

driver.quit()

li_rows



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
finaldf.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/politifact_v1_false.csv")