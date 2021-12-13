# Part A 

import pandas as pd

df = pd.read_csv(r"D:\UCSC\UCSC\NLP 220 - Data Collection, Wrangling and Crowdsourcing\Assignments\Assignment 4\Tweets.csv")

#  Print the total count of data samples.

print("Count of Data Samples:", len(df))

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

airline_sentiment = list(df['airline_sentiment'])
negativereason = list(df['negativereason'])

airline_sentiment_unique = list(set(airline_sentiment))
negativereason_unique = list(set(negativereason))

# For the columns “airline_sentiment” and “negativereason”, determine the number of unique values, as well as the most frequent value and its frequency.
print("Number of Unique 'Airline Sentiment' values: ",len(airline_sentiment_unique))
print("Number of Unique 'Negative Reason' values: ",len(negativereason_unique))

from collections import  Counter

airline_sentiment_fd = Counter(airline_sentiment)
negativereason_fd = Counter(negativereason)

print("\nMost frequent 'Airline Sentiment' and its frequency : ", list(airline_sentiment_fd.most_common(1)[0]))
print("\nMost frequent 'Negative Reason' and its frequency : ", list(negativereason_fd.most_common(1)[0]))

tweets = df['text']

max_tweet = max(tweets, key = len)
min_tweet = min(tweets, key = len)

# Print the lengths of the shortest and the longest tweet in the dataset.

print("\nLongest tweet in the dataset: ",max_tweet) 
print("\nLongest tweet length: ",len(max_tweet))
print("\nShortest tweet in the dataset: ", min_tweet)
print("\nShortest tweet length: ",len(min_tweet))

# Plot the tweet length distribution in the form of a histogram.

import matplotlib.pyplot as plt
import numpy as np

tweet_len = [len(i) for i in tweets]

nearest_multiple = 5 * round(len(max_tweet)/5)

bin = [ j for j in range(1,len(max_tweet),5)]

fig, ax = plt.subplots(1, 1)

ax.set_title("Tweet length distribution")
ax.set_xlabel('Tweet length')
ax.set_ylabel('Number of tweets')

# print(bin)
ax.hist(tweet_len, bins= bin )
plt.show()

"""#Part B """

df2 = df[['airline','airline_sentiment']]

df_virgin = df2[df['airline'] == 'Virgin America']
df_american = df2[df['airline'] == 'American']
df_Delta = df2[df['airline'] == 'Delta']
df_Airways = df2[df['airline'] == 'US Airways']
df_United = df2[df['airline'] == 'United']
df_Southwest = df2[df['airline'] == 'Southwest']

df_Southwest = df_Southwest.groupby(['airline_sentiment']).count()
df_Southwest= df_Southwest.rename(columns={"airline": "Southwest"})
df_virgin = df_virgin.groupby(['airline_sentiment']).count()
df_virgin = df_virgin.rename(columns={"airline": "Virgin"})
df_american = df_american.groupby(['airline_sentiment']).count()
df_american = df_american.rename(columns={"airline": "American"})
df_Delta = df_Delta.groupby(['airline_sentiment']).count()
df_Delta = df_Delta.rename(columns={"airline": "Delta"})
df_Airways = df_Airways.groupby(['airline_sentiment']).count()
df_Airways = df_Airways.rename(columns={"airline": "US Airways"})
df_United = df_United.groupby(['airline_sentiment']).count()
df_United = df_United.rename(columns={"airline": "United"})

df_final = pd.concat([df_Southwest, df_virgin, df_american, df_Delta, df_Airways, df_United], axis=1)
df_final

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    colors = ['orange','green','red']
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        my_colors = ['green','red','orange']
        df[var_name].plot(kind='bar', stacked=True, color=my_colors)
        
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()
    fig.set_size_inches(20.5, 10.5)
    plt.show()

draw_histograms(df_final, df_final.columns, 2, 3)

""" #part C """

import re

def tokenize(str):
  
  split_pattern = rf"(\w+')(?:\W+|$)|('\w+)|(?:\s+)|(\W)"

  tokens = [x for x in re.split(r"([^'\w\s]|'(?![^\W\d_])|(?<![^\W\d_])')|(?='(?<=[^\W\d_]')(?=[^\W\d_]))|\s+", str) if x]

  return tokens


tweets = list(df['text'])

tokenized_tweets = [tokenize(i) for i in tweets]

print(tokenized_tweets)

"""# Part D"""

import nltk
nltk.download('punkt')

df_tokenized = pd.DataFrame(columns=['NLTK', 'Sriram'])

# Uncomment below if you want to run and display differences on all tweets text

# for i in tweets:
#   df_tokenized.loc[len(df_tokenized)] = [nltk.word_tokenize(i), tokenize(i)]
# print(df_tokenized)



# Print the differences for 5 examples where your tokenizer behaves differently from the NLTK word tokenizer. 

count = 0

for i in tweets:
  if nltk.word_tokenize(i) !=  tokenize(i):
    print("\n NLTK Tokenized Tweet #",count,":",nltk.word_tokenize(i) )
    print("Custom Tokenized Tweet #",count,":",tokenize(i) )
    count +=1
  if count== 6:
    break

"""# Part E"""

# Find the number of missing values for tweet-location and user_timezone field. Drop the missing values using Pandas.

location_NaN = len(df[df['tweet_location'].isna() == True])
timezone_NaN = len(df[df['user_timezone'].isna() == True])

print("number of missing values for tweet-location", location_NaN)
print("number of missing values for user_timezone", timezone_NaN)

df = df[df['tweet_location'].isna() == False]
df = df[df['user_timezone'].isna() == False]
df.reset_index(inplace=True,drop=True)

# Now look at the tweet_created field. When you parse the file using Python, do you see this as parsed as date or a string? Write the code to properly parse this as date.

df['tweet_created']= pd.to_datetime(df['tweet_created'])

for j in df['tweet_created'].head():
  print(j, " ---> is now --->", type(j))

# Find the total number of tweets which are from Philadelphia (Think about misspelling, think about spaces between characters). Find all different spellings of Philadelphia.

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

locations = list(set(list(df['tweet_location'])))
location_list_ph = process.extract("Philadelphia PA", locations, limit=15)
location_list_ph = [i[0] for i in location_list_ph if len(i[0]) > 2]

print(df[df['tweet_location'].isin(location_list_ph)])
print("Number of tweets from Philadelphia: ", len(df[df['tweet_location'].isin(location_list_ph)]))

"""# Part F """

# Simple random sampling

length = len(df)

simple_train = df.sample(n = round(length * 0.7) )
simple_dev =  df.sample(n = round(length * 0.2) )
simple_test =  df.sample(n = round(length * 0.1) )

print("Number of examples - random sampling - Train set: ",len(simple_train))
simple_train.to_csv("simple-train.csv")
print("Number of examples - random sampling - Dev set: ",len(simple_dev))
simple_dev.to_csv("simple-dev.csv")
print("Number of examples - random sampling - Test set: ",len(simple_test))
simple_test.to_csv("simple-test.csv")

# Stratified random sampling

from sklearn.model_selection import StratifiedShuffleSplit

train_straified = StratifiedShuffleSplit(n_splits=1, test_size= round(length * 0.7))
dev_straified = StratifiedShuffleSplit(n_splits=1, test_size= round(length * 0.2))
test_straified = StratifiedShuffleSplit(n_splits=1, test_size= round(length * 0.1))

for x,y in train_straified.split(df,df['airline']):
  Stratified_train = df.iloc[y, :]
print("Number of examples -  Stratified random sampling - Train set: ",len(Stratified_train))
Stratified_train.to_csv("stratified-train.csv")

for x,y in dev_straified.split(df,df['airline']):
  Stratified_dev = df.iloc[y, :]
print("Number of examples -  Stratified random sampling - Dev set: ",len(Stratified_dev))
Stratified_dev.to_csv("stratified-dev.csv")

for x,y in test_straified.split(df,df['airline']):
  Stratified_test = df.iloc[y, :]
print("Number of examples - Stratified random sampling - Test set: ",len(Stratified_test))
Stratified_test.to_csv("stratified-test.csv")

# Distributions of labels for each set

print("Simple train: ", dict(Counter(simple_train['airline_sentiment'])))

print("Simple dev: ", dict(Counter(simple_dev['airline_sentiment'])))

print("Simple test: ", dict(Counter(simple_test['airline_sentiment'])))

print("\nStratified train: ", dict(Counter(Stratified_train['airline_sentiment'])))

print("Stratified dev: ", dict(Counter(Stratified_dev['airline_sentiment'])))

print("Stratified test: ", dict(Counter(Stratified_test['airline_sentiment'])))

"""# Part G"""

# Use Twitter’s public API to crawl 500 tweets containing the keyword “covid-19” and “vaccination rate”. 

import twitter
import json
from urllib.parse import unquote

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_TOKEN_SECRET = ''

auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)

twitter_api = twitter.Twitter(auth=auth)

q = '"covid-19 vaccination rate"' 

count = 500

search_results = twitter_api.search.tweets.all(q=q, count=count)  # Add Keys and secrets if you get error here.

statuses = search_results['statuses']

for _ in range(5):
    print('Length of statuses', len(statuses))
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError as e:
        break
        
    kwargs = dict([ kv.split('=') for kv in unquote(next_results[1:]).split("&") ])
    
    search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']


print(json.dumps(statuses[0], indent=1))
print(json.dumps(statuses[2], indent=1))
print(json.dumps(statuses[3], indent=1))

with open ('tweets.json', mode='w+', encoding='UTF-8') as f:
    f.write(json.dumps(statuses, indent=3))

with open ("tweets.json", mode='r+', encoding='UTF-8') as f2:
  tweets_json = json.load(f2)

for i in range(0,len(tweets_json)):
  print(tweets_json[i]['text'])

# Output the tweets in a CSV file.

import json

with open (r"D:\UCSC\UCSC\NLP 220 - Data Collection, Wrangling and Crowdsourcing\Assignments\Assignment 4\tweets.json", mode='r+', encoding='UTF-8') as f2:
  tweets_json = json.load(f2)

import pandas as pd

df_tweets = pd.json_normalize(tweets_json)
df_tweets = df_tweets['text']
print("Displaying just 5 values from Tweets collected: \n", df_tweets[:5])

# Store the tweets to a database of your choice.

from sqlalchemy import create_engine
import sqlite3

sqlite3.connect('tweetsDB.db')

engine = create_engine('sqlite:///tweetsDB.db', echo=True)

df_tweets.to_sql('tweets', con=engine)  # Table exists error means the db has already been provided. Please change db name in line 317 and 319.


# Tweet text into n-grams  (1-gram, 2-gram and 3-gram) using own tokenizer

import re

def my_ngrams(s, n):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

from itertools import chain

bigrams = []
trigrams = []
unigrams = []

for i in df_tweets:
  unigrams.append(my_ngrams(i,1))
  bigrams.append(my_ngrams(i,2))
  trigrams.append(my_ngrams(i,3))

unigrams = list(chain.from_iterable(unigrams))
bigrams = list(chain.from_iterable(bigrams))
trigrams = list(chain.from_iterable(trigrams))

# Frequency distribution
unigram_fd = dict(Counter(unigrams).most_common(len(unigrams)))
bigrams_fd = dict(Counter(bigrams).most_common(len(bigrams)))
trigrams_fd = dict(Counter(trigrams).most_common(len(trigrams)))

print("Frequency distribution of Unigrams: ", unigram_fd )
print("Frequency distribution of Bigrams: ", bigrams_fd)
print("Frequency distribution of Trigrams: ", trigrams_fd )

# Frequency distribution visual for unigrams
unigram_fd = dict(Counter(unigrams).most_common(10)) # Displaying top 10 DUE TO VISUAL SPACE CONSTRAINT

lists = sorted(unigram_fd.items())
fig1 = plt.figure()
x, y = zip(*lists)
fig1.set_size_inches(20.5, 10.5)
plt.bar(x, y)
plt.show()

# Frequency distribution visual for bigrams
bigrams_fd = dict(Counter(bigrams).most_common(10)) # Displaying top 10 DUE TO VISUAL SPACE CONSTRAINT

lists = sorted(bigrams_fd.items())
fig1 = plt.figure()
x, y = zip(*lists)
fig1.set_size_inches(20.5, 10.5)
plt.bar(x, y)
plt.show()

# Frequency distribution visual for trigrams
trigrams_fd = dict(Counter(trigrams).most_common(10)) # Displaying top 10 DUE TO VISUAL SPACE CONSTRAINT

lists = sorted(trigrams_fd.items())
fig1 = plt.figure()
x, y = zip(*lists)
fig1.set_size_inches(20.5, 10.5)
plt.bar(x, y)
plt.show()

# ! THANK YOU PROFESSOR !