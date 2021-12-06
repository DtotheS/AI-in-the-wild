import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.layers import LSTM

os.getcwd()
df = pd.read_csv("data_associative_inference - Sheet1.csv")
df.columns
d_fake = df['published_date']

df = df.dropna()

type(d_fake)

cl = df['claim']
c1 = df['content1']
c2 = df['content2']
c3 = df['content3']
c4 = df['content4']
c5 = df['content5']
c6 = df['content6']
c7 = df['content7']

contents = df[['claim','content1','content2','content3','content4','content5','content6','content7']]
contents

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
corpus = []
words = []
for i in range(0,len(contents)):
    review = re.sub('[^a-zA-Z0-9]',' ',df['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    statements = ' '.join(review)
    corpus.append(statements)
    words.append(review)
