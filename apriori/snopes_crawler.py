## Set up
# !brew install chromedriver
# !conda install selenium
# !conda install beautifulsoup4

## Modules
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import csv

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


# For the snopes.factcheck source (source ==1)
# From each url of fact check webpage, get sources_num,category,type,legitimacy,source,url,title,date,claim,sources
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

    date = driver.find_element_by_tag_name("time").get_attribute("datetime")
    claim = driver.find_element_by_class_name("claim-text").text
    legitimacy = driver.find_element_by_class_name("h3").text
    body = driver.find_element_by_class_name("single-body").text
    try: # For the cases which do not contain sources
        body = body.split("Sources:")
        body = body[1]
        body = body.split("\n")
    except:
        body = []
    sources = [x for x in body if x]
    sources_num = len(sources)

    return sources_num,"politics","fact_check",legitimacy,source,url,title,date,claim,sources

### Start to Collect Data

options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',options=options)

start_link = 'https://www.snopes.com/fact-check/category/politics/'
urls = get_urls(start_link,2) # start_link and number of pages you want to collect.

# Collect Data from each fact-check webpage and write in CSV.
header = ['id','sources_num','category','type','legitimacy','source_name','url','title','date','text','sources']
collect_csv = "/Users/agathos/DtotheS/AI-in-the-wild/apriori/snopes_data.csv"
with open(collect_csv,'w') as f:
    c = csv.writer(f) # write csv on f.
    c.writerow(header) # header
    i = 1
    for link in urls:
        foo = snopes_picker(link)
        foo = list(foo)
        foo.insert(0,i)
        c.writerow(foo)
        i += 1

driver.close()
# TODO
# Solved: 1. Go to the large dataset: next page
# 2. Sources parsing and add as data
# 3. Collect text for each source
# Solved: does not have sources case: https://www.snopes.com/fact-check/chanel-poop-building/
