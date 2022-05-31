## Set up
# !brew install chromedriver
# !conda install selenium

## Modules
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import csv
import time

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
    wait = WebDriverWait(driver, 10)
    smg = driver.execute_script('return smg') # execute javascript var "smg"

    # Title
    try:
        title = smg['page_data']['title']
    except:
        try:
            title_long = driver.title
            title_long = title_long.split("|")
            title = title_long[0] + "\n" + driver.find_element_by_class_name("subtitle").text  # title + subtitle
        except:
            title = driver.title + "\n" + driver.find_element_by_class_name("subtitle").text

    try:
        content_owner = smg['page_data']['content_owner']
    except:
        try:
            content_owner = title_long[1]
        except:
            content_owner = 'Snopes'

    try:
        date_published = smg['page_data']['date_published']
    except:
        try:
            da = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_tag_name("time")
            dates = [i.get_attribute("datetime") for i in da]
            date_published = dates[0]
        except:
            date_published = "none"

    try:
        date_updated = smg['page_data']['date_updated']
    except:
        try:
            date_updated = dates[1:]
        except:
            date_updated = "none"

    try:
        primary_category = smg['page_data']['primary_category']
    except:
        primary_category = "none"

    try:
        rating = smg['page_data']['rating']
    except:
        try:
            rating = driver.find_element_by_class_name("h3").text
        except:
            rating = "none"

    try:
        tags = smg['page_data']['tags']
    except:
        tags = "none"

    try:
        author_name = smg['page_data']['author_name']
    except:
        try:
            au = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors")
            author_name = [i.text for i in au]
        except:
            author_name = "none"


    try:
        claim = driver.find_element_by_class_name("claim-text").text
    except:
        claim = "none"

    # Bodyt = body text & claim part
    foo = driver.find_elements_by_css_selector("div.list-group-item.small")
    if foo:
        srcs = []
        for i in foo:
            srcs.append(i.text)
        bodyt = driver.find_element_by_class_name("single-body").text
    else:
        try: # For the cases which do not contain sources
            body = driver.find_element_by_class_name("single-body").text
            body = body.split("Sources:")
            bodyt = body[0]
            srcs = body[1]
            srcs = srcs.split("\n")
        except:
            try:
                body = driver.find_element_by_class_name("single-body").text
                body = body.split("Sources")
                bodyt = body[0]
                srcs = body[1]
                srcs = srcs.split("\n")
            except:
                srcs = []
                bodyt = driver.find_element_by_class_name("single-body").text
    sources = [x for x in srcs if x]
    sources_num = len(sources)
    sourceid = 0

    return title,url,claim,rating,content_owner,author_name,date_published,date_updated,primary_category,tags,sourceid,sources_num,sources,bodyt

'''
def snopes_picker(url):
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    try:
        title_long = driver.title
        title_long = title_long.split("|")
        title = title_long[0] + "\n" + driver.find_element_by_class_name("subtitle").text  # title + subtitle
        source = title_long[1]
    except:
        title = driver.title + "\n" + driver.find_element_by_class_name("subtitle").text
        source = "Snopes.com"

    try:
        au = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors")
        authors = [i.text for i in au]
        au_link = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors [href]")
        authors_link = [i.get_attribute("href") for i in au_link]
    except:
        authors = "none"
        authors_link = "none"

    try:
        da = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_tag_name("time")
        dates = [i.get_attribute("datetime") for i in da]
    except:
        dates = "none"

    try:
        claim = driver.find_element_by_class_name("claim-text").text
    except:
        claim = "none"

    try:
        legitimacy = driver.find_element_by_class_name("h3").text
    except:
        legitimacy = "none"

    # Bodyt = body text & claim part
    foo = driver.find_elements_by_css_selector("div.list-group-item.small")
    if foo:
        srcs = []
        for i in foo:
            srcs.append(i.text)
        bodyt = driver.find_element_by_class_name("single-body").text
    else:
        try: # For the cases which do not contain sources
            body = driver.find_element_by_class_name("single-body").text
            body = body.split("Sources:")
            bodyt = body[0]
            srcs = body[1]
            srcs = srcs.split("\n")
        except:
            try:
                body = driver.find_element_by_class_name("single-body").text
                body = body.split("Sources")
                bodyt = body[0]
                srcs = body[1]
                srcs = srcs.split("\n")
            except:
                srcs = []
                bodyt = driver.find_element_by_class_name("single-body").text
    sources = [x for x in srcs if x]
    sources_num = len(sources)
    sourceid = 0

    return sourceid,sources_num,"politics","fact_check",legitimacy,source,url,title,dates,bodyt,claim,sources,authors,authors_link
'''

### Start to Collect Data
options = webdriver.ChromeOptions()
options.add_argument('--headless') # A Headless browser runs in the background. You will not see the browser GUI or the operations been operated on it.
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")

driver = webdriver.Chrome('chromedriver',options=options)

'''
driver.get("https://www.snopes.com/fact-check/ca-gop-ballot-drop-boxes/")
# date = driver.find_elements_by_tag_name("time")

# find dates (including updated date), author name, links example
authors = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors")[0].text
authors_link = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_class_name("authors [href]")[0].get_attribute("href")
dates = driver.find_elements_by_class_name("single-header.border.mb-3")[0].find_elements_by_tag_name("time")
dates[0].get_attribute("datetime")
dates[1].get_attribute("datetime")
'''

'''
# Collect new urls from start_link page.
start_link = 'https://www.snopes.com/fact-check/category/politics/'
urls = get_urls(start_link,100) # start_link and number of pages you want to collect.

# Save urls into csv file.
import csv
header_url = ['id','url']
i = 0
with open("./AI-in-the-wild/apriori/1218_100pgs.csv","w") as f:
    write = csv.writer(f)
    write.writerow(header_url)
    for ele in urls:
        i += 1
        foo2 = [i,ele]
        write.writerow(foo2) # need to make a list. Otherwise, each character will be save into each column.
'''

# Collect Date: May 30 (Mon.) 2022, 11:17AM.
s = 1 # start page number
e = 100 # end page number
urls_f = []
for i in range(1300//100):
    urls = get_urls_by_num(s,e)
    urls_f.extend(urls)
    print(f"{e}" + "at: ", time.localtime())
    s += 100
    e += 100
    time.sleep(600)
driver.close()

# Check duplicates: duplicates, which is page error on their side. e.g.(morsi-you-in-court) https://www.snopes.com/fact-check/page/1078/
for ele in urls_f:
    if urls_f.count(ele) > 1:
        print(ele)

urls_set = [i for n,i in enumerate(urls_f) if i not in urls_f[:n]]
len(urls_f) - len(urls_set)

# Save urls into csv file.
import csv
header_url = ['id','url']
i = 0
with open("./AI-in-the-wild/apriori/urls_053022.csv","w") as f:
    write = csv.writer(f)
    write.writerow(header_url)
    for ele in urls_set:
        i += 1
        foo2 = [i,ele]
        write.writerow(foo2) # need to make a list. Otherwise, each character will be save into each column.

## Read urls from csv file.
# import pandas as pd
# df = pd.read_csv("./AI-in-the-wild/apriori/1218_100pgs.csv") #100 urls + 3 cases of our stimuli (another 1 is from politifacts)
# urls = df['url'].values.tolist()

# Collect Data from each fact-check webpage and write in CSV.
header = ['id','title','url','claim','rating','content_owner','author_name','date_published','date_updated','primary_category','tags','sourceid','sources_num','sources','bodyt']
collect_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/fcs_053022.csv"
with open(collect_csv,'w',encoding="utf-8") as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    i = 1
    for link in urls_set[0:99]:
        foo = snopes_picker(link)
        print(link)
        foo = list(foo)
        foo.insert(0,i)
        c.writerow(foo)
        i += 1
driver.close()

'''
# I do not know why, but it only save 1182 urls....
txtfile = open("./AI-in-the-wild/apriori/1218_100pgs.csv","w")
for ele in urls:
    txtfile.write(ele + "\n")
'''

'''
## Save urls into csv file.
import csv
header_url = ['id','url']
i = 0
with open("./AI-in-the-wild/apriori/1218_100pgs.csv","w") as f:
    write = csv.writer(f)
    write.writerow(header_url)
    for ele in urls:
        i += 1
        foo2 = [i,ele]
        write.writerow(foo2) # need to make a list. Otherwise, each character will be save into each column.

## Read urls from csv file
import pandas as pd
df = pd.read_csv("./AI-in-the-wild/apriori/1218_100pgs.csv")
urls = df['url'].values.tolist()

# with open("./AI-in-the-wild/apriori/1218_100pgs_noid.csv","r") as ff: 
#     urls = [line.rstrip("\n") for line in ff]
'''

# TODO
# Solved: 1. Go to the large dataset: next page
# Solved: 2. Sources parsing and add as data
# half: 3. Collect text for each source
# Solved: does not have sources case: https://www.snopes.com/fact-check/chanel-poop-building/





