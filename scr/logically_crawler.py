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
# driver.get("https://www.logically.ai/factchecks/library/page/1")
# https://www.logically.ai/factchecks/library/tag/loc-us/page/83 #US
# https://www.logically.ai/factchecks/library/tag/loc-uk/page/25 #UK
# https://www.logically.ai/factchecks/library/tag/loc-in/page/118 #India
# https://www.logically.ai/factchecks/library/page/364 # all
# driver.current_url
# driver.close()
rs = []
ds = []
ts = []
lis = []
pages = range(1,365)
for i in pages:
    cpg = "https://www.logically.ai/factchecks/library/page/%s" %i
    driver.get(cpg)
    boxes = driver.find_elements_by_class_name("grid-item-4")
    for k in range(len(boxes)):
        try:
            rs.append(boxes[k].find_element_by_class_name("fc-verdict").text)  # rating
        except:
            rs.append(None)
        try:
            ds.append(boxes[k].find_element_by_tag_name("p").text)  # date
        except:
            ds.append(None)
        try:
            ts.append(boxes[k].find_element_by_tag_name("h3").text)  # title
        except:
            ts.append(None)
        try:
            lis.append(boxes[k].find_element_by_tag_name("a").get_attribute("href"))  # link
        except:
            lis.append(None)
    print(str(i) +" "+cpg +" Done")

len(rs) == len(ds) == len(ts) == len(lis)

df = pd.DataFrame()
df['link']=lis
df['site']="logically"
df['title'] = ts
df['rating'] = rs
df['fc_date'] = ds

df['fc_date'] = pd.to_datetime(df['fc_date'],format="%d/%m/%Y")
df['fc_year'] = df.apply(lambda row : row['fc_date'].year, axis=1)
df['fc_month'] = df.apply(lambda row : row['fc_date'].month, axis=1)
df['fc_day'] = df.apply(lambda row : row['fc_date'].day, axis=1)

df['fc_year'].isnull().sum()
df['fc_month'].isnull().sum()
df['fc_day'].isnull().sum()

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/logically_090622.csv")

df[['author','src_num','src_name','src_rating','src_link','key_word','location','body']] = None
for i in range(len(df)):
    driver.get(df['link'][i])
    try:
        df['author'][i] = driver.find_element_by_class_name("fc-posted-by").text #author
    except:
        pass

    try:
        refs = driver.find_element_by_class_name("reference-links").find_elements_by_class_name("flex-grid")
        refs_links = driver.find_element_by_class_name("reference-links").find_elements_by_tag_name("a")
        df['src_num'][i] = len(refs)
        df['src_name'][i] = []
        df['src_rating'][i] = []
        df['src_link'][i] = []
        for k in range(len(refs)):
            df['src_name'][i].append(refs[k].text.strip().split("\n")[0]) # source
            df['src_rating'][i].append(refs[k].text.strip().split("\n")[1]) # source rating
            df['src_link'][i].append(refs_links[0].get_attribute("href")) # source link
    except:
        pass
    try:
        kwords = driver.find_elements_by_class_name("factcheck-post__tags")
        df['key_word'][i] = []
        for w in range(len(kwords)):
            df['key_word'][i].append(kwords[w].text) #tags
    except:
        pass

    try:  # location info
        driver.find_element_by_id("hs_cos_wrapper_location_icon")
    except:
        pass
    else:
        df['location'][i] = driver.find_element_by_class_name("fc-post-meta").find_element_by_class_name("with-padding-small").text

    try:
        df['body'][i] =driver.find_element_by_class_name("blog-post__body").text #body
    except:
        pass

    print(str(i) + " " + df['link'][i] +" Done")

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/logically_090622_v2.csv",index=False)