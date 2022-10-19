# !pip install tweepy --upgrade
# token: AAAAAAAAAAAAAAAAAAAAAMyudQEAAAAA7Ht0xE50GHbURXiAITeSvtlnZ4M%3DprJeuH9Y73bmrDJ6fOXXH268XJEqPKbTdgNIM39RjWwvoAI96U
''' References
Building Query: https://developer.twitter.com/en/docs/twitter-api/tweets/counts/integrate/build-a-query
APIv2 to PD: https://www.kirenz.com/post/2021-12-10-twitter-api-v2-tweepy-and-pandas-in-python/twitter-api-v2-tweepy-and-pandas-in-python/
Code sample: https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9
API reference index: https://developer.twitter.com/en/docs/api-reference-index
'''
import tweepy
from zipfile import ZipFile
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from datetime import datetime as dt
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAMyudQEAAAAA7Ht0xE50GHbURXiAITeSvtlnZ4M%3DprJeuH9Y73bmrDJ6fOXXH268XJEqPKbTdgNIM39RjWwvoAI96U',wait_on_rate_limit=True)

os.chdir("/Users/agathos/DtotheS/AI-in-the-wild/scr")
sn = pd.read_csv("../data/sn_100222.csv")

## Search for Tweets containing specific url
print(dt(2006,3,22))
i = 1
sn['link'][1]
sn['likes'] = 0
sn['retweets'] = 0
sn['tot'] = 0
for i in range(5):
    try:
        query = 'from:snopes url:"{}"'.format(sn['link'][i])
        print(query)
        tweets = client.search_all_tweets(query=query, tweet_fields=['context_annotations', 'created_at', 'id'],max_results=100, start_time=dt(2006,3,22))
        tweets
        len(tweets.data)
        sum_likes = 0
        sum_rtweets = 0
        for tweet in tweets.data:
            id = tweet.id
            users = client.get_liking_users(id=id, user_fields='username')  # currently, APIv2 only give users who liked within a month, so there is a mismatch to end_point vs APIv2 data
            rusers = client.get_retweeters(id=id, user_fields='username')
            sum_likes += users.meta['result_count']
            sum_rtweets += rusers.meta['result_count']
        sn['likes'][i] = sum_likes
        sn['retweets'][i] = sum_rtweets
        sn['tot'][i] = sum_likes + sum_rtweets
        print(i,": ", sum_likes + sum_rtweets)
    except:
        print(i,"something wrong")





## 1. Searching for Tweets from the last 7 days
# Replace with your own search query
query = 'from:suhemparack -is:retweet'

tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

for tweet in tweets.data:
    print(tweet.text)
    if len(tweet.context_annotations) > 0:
        print(tweet.context_annotations)


## 2. Searching for Tweets from the full-archive of public Tweets
# Replace with your own search query
query = 'from:snopes -is:retweet https://www.snopes.com/fact-check/gas-prices-under-trump/'

tweets = client.search_all_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

for tweet in tweets.data:
    print(tweet.text)
    if len(tweet.context_annotations) > 0:
        print(tweet.context_annotations)

## 3. Getting Tweets from the full-archive of public Tweets for a specific time-frame
# Replace with time period of your choice
start_time = '2020-01-01T00:00:00Z'
# Replace with time period of your choice
end_time = '2020-08-01T00:00:00Z'

tweets = client.search_all_tweets(query=query, tweet_fields=['context_annotations', 'created_at'],
                                  start_time=start_time,
                                  end_time=end_time, max_results=100)

for tweet in tweets.data:
    print(tweet.text)
    print(tweet.created_at)

## 4. Getting more than 100 Tweets at a time using paginator

query = 'covid -is:retweet'
# Replace the limit=1000 with the maximum number of Tweets you want
for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=1000):
    print(tweet.id)
# response = client.get_tweets(ids=[1533928308890603521])

## 5. Writing Tweets to a text file
# Replace with your own search query
query = 'covid -is:retweet'

# Name and path of the file where you want the Tweets written to
file_name = 'tweets.txt'

with open(file_name, 'a+') as filehandle:
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                                  tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(
            limit=1000):
        filehandle.write('%s\n' % tweet.id)
