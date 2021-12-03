"""By HaeseungSeo 11/10/2019"""

import pandas as pd
import string
from nltk.corpus import stopwords
import csv
import nltk



df1=pd.read_csv('/Users/vitamind/Documents/test_tweet/FINALCHECK/final_combined_tweets/47_combined.csv')

UniqueTweet = df1['friend_tweet'].unique().astype(str)
date=df1['date']


essentialA =[]
essential99 = []
essential100 = []
date100 = []


for i, line in enumerate(UniqueTweet):
   if "hot dog" in line:
      essentialA.append(line)

      alphabet1 = [c for c in line if c not in string.punctuation]
      joined1 = ''.join(alphabet1)
      essential1 = [word for word in joined1.split() if word.lower() not in stopwords.words('english')]

      tags = nltk.pos_tag(essential1)
      nouns = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]


      essential11 = ' '.join(nouns)


      stopword = {'hot dog', 'NYC', 'new york', 'New York'}
      resultwords1 = [word for word in nouns if word not in stopword]
      essential11 = ' '.join(resultwords1).lower()

      essential99.append(essential1)
      essential100.append(essential11)
      date100.append(date[i])


essentialB=[]
essential199 =[]
essential200 =[]
date200 = []
for i, line in enumerate(UniqueTweet):
    if "NYC" in line:
        essentialB.append(line)

        alphabet2 = [c for c in line if c not in string.punctuation]
        joined2 = ''.join(alphabet2)
        essential2 = [word for word in joined2.split() if word.lower() not in stopwords.words('english')]

        tags = nltk.pos_tag(essential2)
        nouns = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        essential22 = ' '.join(nouns)

        stopword = {'hot dog', 'NYC', 'new york', 'New York'}
        resultwords2 = [word for word in nouns if word not in stopword]
        essential22 = ' '.join(resultwords2).lower()

        essential199.append(essential2)
        hh2 = essential200.append(essential22)
        date200.append(date[i])


with open ('New_test_matched__47_combined.csv','w+') as file:
    writer =csv.writer(file,delimiter =',',quotechar ='"')
    writer.writerow(['A_tweet','date1','C_tweet','date2','Common_word'])

    matched_dict = {}

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
            essentialA[essential100.index(line[0])],
            line[2],
            essentialB[essential200.index(line[1])],
            line[3],
            matched_dict[line]])