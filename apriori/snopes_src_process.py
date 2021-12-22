import re
from dateutil.parser import parse
import pandas as pd
import ast
from dateparser.search import search_dates
from datetime import datetime as dt
import csv

data = pd.read_csv("/Users/agathos/DtotheS/AI-in-the-wild/apriori/aidata_1203fc.csv",on_bad_lines='skip')
maxid = int(max(data['id']))

# data.columns
# srcdf = pd.DataFrame(columns=['id','sourceid','source_num','category','type','legitimacy','source_name','url','title','date','text'])
with open(r"/Users/agathos/DtotheS/AI-in-the-wild/apriori/aidata_1203fc.csv",'a',encoding="utf-8") as f:
    c = csv.writer(f) # write csv on f.
    c.writerow("\n")
    for i in range(maxid):
        x = data["sources"][i]
        x = ast.literal_eval(x) # list of sources
        x = [n.strip() for n in x]
        for l in range(len(x)):
            id = i+1
            sourceid = l+1
            category = "politics-related"
            type = 'source'
            legitimacy = "none"
            foo1 = x[l].replace('“', '"').replace('”', '"') # replace left-double quotation(“ is U+201C) and right-double quotation(” is U+201D) with " which is U+0022.
            # text[0][16].encode("unicode_escape")
            try:
                title = re.findall(r'"([^"]*)"', foo1)
                title = title[0]
            except:
                title = "none"
            try:
                date = search_dates(foo1)
                date = date[-1][-1]
                date = "{}/{}/{}".format(date.month, date.day, date.year)
            except:
                date = "none"
            try:
                url = re.findall(r'(https?://[^\s]+)', foo1)
                url = url[0]
            except:
                url = "none"
            source_name = "none"
            source_num = "none"
            row = [id,sourceid,source_num,category,type,legitimacy,source_name,url,title,date]
            c.writerow(row)

#### Test Field
'''
sources[0]
for i in range(maxid):
    x = data["sources"][0]
    x = ast.literal_eval(x) # list of sources
    x = [n.strip() for n in x]
foo1 = x[l].replace('“', '"').replace('”', '"')
title = re.findall(r'"([^"]*)"', foo1)
title = title[0]

url = re.findall(r'(https?://[^\s]+)', foo1)
url = url[0]
date = search_dates(foo1)
date = date[-1][-1]
"{}/{}/{}".format(date.month,date.day,date.year)

    for l in range(len(x)):
        id = i
        sourceid = l+1
        category = "politics-related"
        type = 'source'
        legitimacy = "none"
        foo1 = x[l].replace('“', '"').replace('”', '"') # replace left-double quotation(“ is U+201C) and right-double quotation(” is U+201D) with " which is U+0022.
        # text[0][16].encode("unicode_escape")
        try:
            title = re.findall(r'"([^"]*)"', foo1)
        except:
            title = "none"
        try:
            date = search_dates(foo1)
            date = date[-1][-1]
        except:
            date = "none"
        try:
            url = re.findall(r'(https?://[^\s]+)', foo1)
        except:
            url = "none"
        source_name = "none"
        source_num = ""
        row = [id,sourceid,source_num,category,type,legitimacy,source_name,url,title,date]
        c.writerow(row)
'''