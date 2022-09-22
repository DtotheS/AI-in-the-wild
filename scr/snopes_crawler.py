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

# driver = webdriver.Chrome(ChromeDriverManager().install())
# driver.close()

## Define Functions

# Get urls of fact check webpages
def get_urls(starting_page, num_pages):
    i = 1
    driver.get(starting_page)
    wait = WebDriverWait(driver, 10)
    mainbox = driver.find_elements_by_class_name("list-archive [href]") # Do not contains Top10 links.

    urls = []
    for item in mainbox:
        url = item.get_attribute("href")
        urls.append(url)
    next = urls[-1] # the last url indicate the next page.
    del urls[-1]
    i += 1

    while i <= num_pages:
        driver.get(next)
        wait = WebDriverWait(driver, 10)
        mainbox = driver.find_elements_by_class_name("list-archive [href]")

        for item in mainbox:
            url = item.get_attribute("href")
            urls.append(url)
        next = urls[-1]
        del urls[-2:] # from the 2nd page, there are previous, next links. So, we need to remove the last two from the urls.
        i += 1

    return urls

# Get urls of fact check webpages by only using page numbers in fact check section (without topic)
def get_urls_by_num(starting_page, num_pages):
    i = starting_page
    driver.get(f"https://www.snopes.com/fact-check/page/{i}/")
    wait = WebDriverWait(driver, 10)
    mainbox = driver.find_elements_by_class_name("list-archive [href]") # Do not contains Top10 links.

    urls = []
    for item in mainbox:
        url = item.get_attribute("href")
        urls.append(url)
    if i == 1:
        del urls[-1]
    else:
        del urls[-2:] # from the 2nd page, there are hrefs of "previous" and "next". So, we need to remove the last two hrefs from the urls.
    i += 1

    while i <= num_pages:
        driver.get(f"https://www.snopes.com/fact-check/page/{i}/")
        wait = WebDriverWait(driver, 10)
        mainbox = driver.find_elements_by_class_name("list-archive [href]")

        for item in mainbox:
            url = item.get_attribute("href")
            urls.append(url)
        del urls[-2:]
        i += 1
    return urls

# For the snopes.factcheck source (source ==1)
# From each url of fact check webpage, get sources_num,category,type,legitimacy,source,url,title,date,claim,sources

def snopes_picker(url):
    driver.get(url)
    try:
        smg = driver.execute_script('return smg') # execute javascript var "smg"
    except:
        print("smg Error: " + url)
        pass

    # Title
    try:
        title = driver.find_element_by_class_name("title").text
    except:
        try:
            title_long = driver.title
            title_long = title_long.split("|")
            title = title_long[0]
        except:
            title = driver.title

    # Subtitle
    try:
        stitle = driver.find_element_by_class_name("subtitle").text
    except:
        try:
            stitle = smg['page_data']['title']
        except:
            stitle = None


    try:
        content_owner = smg['page_data']['content_owner']
    except:
        content_owner = 'Snopes'

    try:
        date_published = driver.find_elements_by_css_selector("time")[0].text
        # date_published = smg['page_data']['date_published']
    except:
        try:
            date_published = smg['page_data']['date_published']
            # da = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_tag_name("time")
            # dates = [i.get_attribute("datetime") for i in da]
            # date_published = dates[0]
        except:
            date_published = None

    try:
        if len(driver.find_elements_by_css_selector("time")) > 2:
            date_updated = driver.find_elements_by_css_selector("time")[-1].text
        else:
            date_updated = smg['page_data']['date_updated']
    except:
        date_updated = None
        # try:
        #     da = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_tag_name("time")
        #     dates = [i.get_attribute("datetime") for i in da]
        #     date_updated = dates[1:]
        # except:
        #     date_updated = None

    try:
        primary_category = smg['page_data']['primary_category']
    except:
        primary_category = None

    try:
        categories = smg['page_data']['categories']
    except:
        categories = None

    try:
        rating = smg['page_data']['rating']
    except:
        try:
            rating = driver.find_element_by_class_name("h3").text
        except:
            try:
                rating = driver.find_element_by_class_name("claim-old").text
            except:
                try:
                    rating = driver.find_elements_by_css_selector("noindex")[0].text.strip()
                except:
                    rating = None

    try:
        tags = smg['page_data']['tags']
    except:
        tags = None

    try:
        author_name = smg['page_data']['author_name']
    except:
        try:
            au = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors")
            author_name = [i.text for i in au]
        except:
            author_name = None


    try:
        claim = driver.find_element_by_class_name("claim-text").text
    except:
        claim = None

    # Bodyt = body text & claim part
    foo = driver.find_elements_by_css_selector("div.list-group-item.small")
    if foo:
        srcs = []
        for i in foo:
            srcs.append(i.text)
        bodyt = driver.find_element_by_class_name("rich-text").text
    else:
        try: # For the cases which do not contain sources
            body = driver.find_element_by_class_name("rich-text").text
            body = body.split("Sources:")
            bodyt = body[0]
            srcs = body[1]
            srcs = srcs.split("\n")
        except:
            try:
                body = driver.find_element_by_class_name("rich-text").text
                body = body.split("Sources")
                bodyt = body[0]
                srcs = body[1]
                srcs = srcs.split("\n")
            except:
                srcs = []
                try:
                    bodyt = driver.find_element_by_class_name("rich-text").text
                except:
                    bodyt = None

    sources = [x for x in srcs if x]
    sources_num = len(sources)
    sourceid = 0

    try:
        page_type = smg['page_data']['page-type']
    except:
        page_type = None

    return title,claim,rating,content_owner,author_name,date_published,date_updated,primary_category,tags,sourceid,sources_num,sources,bodyt,page_type


### Start to Collect Data
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

driver = webdriver.Chrome('chromedriver',options=options)
# driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)


# 1st Collect Date: May 30 (Mon.) 2022, 11:17AM.
# 2nd Collect Date: Sep 19 (Mon.) 2022, 12:10PM.
s = 1 # start page number
e = 92 # end page number
urls_f = []
for i in range(1300//100): # need to be udpated...
    urls = get_urls_by_num(s,e)
    urls_f.extend(urls)
    print(f"{e}" + "at: ", time.localtime())
    s += 100
    e += 100
    time.sleep(600) # Snopes.com does not allow collect bunch of data at a time, so I gave 10 minutes sleep for every 100 urls.
driver.close()

# Check duplicates: duplicates, which is page error on their side. e.g.(morsi-you-in-court) https://www.snopes.com/fact-check/page/1078/
for ele in urls_f:
    if urls_f.count(ele) > 1:
        print(ele)

urls_set = [i for n,i in enumerate(urls_f) if i not in urls_f[:n]]
len(urls_f) - len(urls_set)

df = pd.DataFrame()
df['url'] = urls_f

'''# You can save the collected urs for future as csv file.
import csv
header_url = ['id','url']
i = 0
with open("./AI-in-the-wild/data/urls_053022.csv","w") as f:
    write = csv.writer(f)
    write.writerow(header_url)
    for ele in urls_set:
        i += 1
        foo2 = [i,ele]
        write.writerow(foo2) # need to make a list. Otherwise, each character will be save into each column.

# urls_set.index("https://wwwsnopes.com/fact-check/prince-philip-prank-queen/")
# urls_set[1488]
'''

# # Read urls from csv file.
# import pandas as pd
# df = pd.read_csv("./AI-in-the-wild/apriori/1218_100pgs.csv") #100 urls + 3 cases of our stimuli (another 1 is from politifacts)
# urls = df['url'].values.tolist()

# Collect Data from each fact-check webpage and write in CSV. #start from 11702
from os.path import exists
import pandas as pd

### Start to Collect Data
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

driver = webdriver.Chrome('chromedriver',options=options)
# driver = webdriver.Chrome(ChromeDriverManager().install(),options=options)

header = ['title','claim','rating','content_owner','author_name','date_published','date_updated','primary_category','tags','sourceid','sources_num','sources','bodyt','page_type']
df[header] = None

for i in range(len(df)):
    link = df['url'][i]
    foo = snopes_picker(link)
    df.loc[i][1:] = foo

import numpy as np
df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_091922.csv",index=False) # further urls were collected.


# csv does not have word limits for each cell. However, excel only can read 32,767 chars. Thus, if, for example, bodyt exceed 32,767 chars, then it may overflow the cell when the csv is opened using excel.
# Below, the code modifed overflew cells.
df2 = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_053022.csv")
df2[df2['content_owner'].isnull()]['id'] #id: 12352, 14054
link = df2[df2['id']==14054].iloc[0]['url']
# df2[df2['id']==14054][header]
foo = snopes_picker(link)
foo = list(foo)
for h,ele in zip(header,foo):
    df2[h][14051] = ele

df2.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_053022.csv",index=False)

'''directly add to csv file
collect_csv = "/Users/agathos/DtotheS/AI-in-the-wild/data/fcs_053022.csv"
if exists(collect_csv) == False:
    with open(collect_csv,'w',encoding="utf-8") as f:
        c = csv.writer(f)  # write csv on f.
        c.writerow(header)  # header
        i = 0  # index number, not id
        for link in urls_set[11702:]:
            if i > 0 and i % 100 == 0: time.sleep(600)
            print(link)
            foo = snopes_picker(link)
            foo = list(foo)
            foo.insert(0, i + 1)
            c.writerow(foo)
            i += 1
else:
    with open(collect_csv, 'a', encoding="utf-8") as f:
        c = csv.writer(f)  # write csv on f.
        try:
            df = pd.read_csv(collect_csv)
            i = len(df)  # index number for new link
        except:
            c.writerow(header) # header
            i = 0  # index number
        for link in urls_set[11702:]:
            if i > 0 and i % 100 == 0: time.sleep(600)
            print(link)
            foo = snopes_picker(link)
            foo = list(foo)
            foo.insert(0, i + 1)
            c.writerow(foo)
            i += 1
driver.close()
'''

# Combined dataset: 0919 + 0530.
cols = df2.columns.to_list()
df = df[cols] # match the order of columns

df.columns == df2.columns
len(df) # 1102
len(df2) # 14390
df = df.append(df2,ignore_index=True)

df.to_csv("/Users/agathos/DtotheS/AI-in-the-wild/data/sn_091922.csv", index=False)