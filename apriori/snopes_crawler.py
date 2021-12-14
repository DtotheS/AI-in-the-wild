# Set up
# !brew install chromedriver
# !conda install selenium
# !conda install beautifulsoup4

#Define getting news function
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
import csv

options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome('chromedriver',options=options)

#Locate the target website
driver.get('https://www.snopes.com/fact-check/')
wait = WebDriverWait(driver, 10)
result = driver.find_elements_by_class_name("stretched-link")


urls = []
for item in result:
    url = item.get_attribute("href") # !!! need to remove top10 results (or duplicates)
    urls.append(url)

# type, legitimacy, source, url, title, date, text(or claim)

# For the snopes.factcheck source (source ==1)
def snopes_picker(url):
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    title_long = driver.title
    title_long = title_long.split("|")
    title = title_long[0] + "\n" + driver.find_element_by_class_name("subtitle").text # title + subtitle
    source = title_long[1]
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
    return sources_num,"fact_check",legitimacy,source,url,title,date,claim,sources

# Collect Data
header = ['id','sources_num','type','legitimacy','source_name','url','title','date','text','sources']
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
# 1. Go to the large dataset: next page
# 2. Sources parsing and add as data
# 3. Collect text for each source
# Solved: does not have sources case: https://www.snopes.com/fact-check/chanel-poop-building/