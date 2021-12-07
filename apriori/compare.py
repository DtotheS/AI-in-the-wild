import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import csv
import nltk


df1=pd.read_csv('/.csv')

FT1=df1.friend_tweet

a1=df1['friend_tweet'].nuniq,.//;.l,kjhg23ue()

a3=np.size(FT1)


UniqueTweet = df1['friend_tweet'].unique().astype(str)

hhhhhh=len(UniqueTweet)


date=df1['date']



essential0 =[]
essential99 = []
essential100 = []
date100 = []

for i, line in enumerate(UniqueTweet):
   if "lunch" in line:
      essential0.append(line)


      alphabet1 = [c for c in line if c not in string.punctuation]
      joined1 = ''.join(alphabet1)
      essential1 = [word for word in joined1.split() if word.lower() not in stopwords.words('english')]




      tagsA = nltk.pos_tag(essential1)
      nounsA = [word for word, pos in tagsA if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]


      stopword = {'lunch','Lunch','LUNCH',"can’t","doesn’t","isn’t","wouldn’t","shouldn’t","didn’t",'…',
                  'i’ve', 'I’ve', 'she’ve', 'She’ve', 'He’ve', 'he’ve', 'They’ve', 'they’ve', 'We’ve', 'we’ve',
                  'Im', "It’s", "it’s", "You’re", "you’re", "I’m", "i’m", "He’s", "She’s", "he’s", "she’s", "They’re",
                  "they’re", "We’re", "we’re",'it', 'thing',
                  "They’ll", "they’ll", "we’ll", "We’ll", "He’ll", "he’ll", "She’ll", "she’ll", "It’ll", "it’ll","I’ll","i’ll",
                  'that','That','this','This','those','Those','These','these',"that’s","That’s"}
      resultwords1 = [word for word in nounsA if word not in stopword]


      essential11 = ' '.join(resultwords1).lower()




      essential99.append(resultwords1)


      essential100.append(line)
      date100.append(date[i])


print("check1")

essential9=[]
essential199 =[]
essential200 =[]
date200 = []
for i, line in enumerate(UniqueTweet):
  if "Trump" in line:
      essential9.append(line)


      alphabet2 = [c for c in line if c not in string.punctuation]
      joined2 = ''.join(alphabet2)
      essential2 = [word for word in joined2.split() if word.lower() not in stopwords.words('english')]

      tags2 = nltk.pos_tag(essential2)
      nouns2 = [word for word, pos in tags2 if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

      essential22 = ' '.join(nouns2)


      stopword = {'Trump','trump','TRUMP',
                  "can’t","doesn’t","isn’t","wouldn’t","shouldn’t","didn’t",'…',
                  'i’ve','I’ve', 'she’ve','She’ve','He’ve','he’ve','They’ve','they’ve','We’ve','we’ve',
                  'Im', "It’s", "it’s", "You’re", "you’re", "I’m", "i’m", "He’s", "She’s", "he’s", "she’s", "They’re",
                  "they’re", "We’re", "we’re",'it', 'thing',
                   "They’ll", "they’ll", "we’ll", "We’ll", "He’ll", "he’ll", "She’ll", "she’ll", "It’ll", "it’ll","I’ll","i’ll",
                  'that','That','this','This','those','Those','These','these',"that’s","That’s"}



      resultwords2 = [word for word in nouns2 if word not in stopword]


      essential22 = ' '.join(resultwords2).lower()


      essential199.append(resultwords2)


      essential200.append(line)
      date200.append(date[i])



with open ('~/.csv','w+') as file:
    writer =csv.writer(file,delimiter =',',quotechar ='"')
    writer.writerow(['A_tweet','date1','C_tweet','date2','Common_word'])

    matched_dict = {}

    if essential99 != essential199:
        for i, sublist1 in enumerate(essential99):
            for j, sublist2 in enumerate(essential199):
                b_match = False
                for val1 in sublist1:
                    for val2 in sublist2:
                        if val1 == val2:
                            b_match = True

                            matched = (essential100[i], essential200[j], date100[i], date200[j])

                            if matched not in matched_dict:
                                matched_dict[matched] = [val1]
                            else:
                                matched_dict[matched].append(val1)
                            break

    for line in list(matched_dict.keys()):
        writer.writerow([
            essential0[essential100.index(line[0])],
            line[2],
            essential9[essential200.index(line[1])],
            line[3],
            matched_dict[line]])
