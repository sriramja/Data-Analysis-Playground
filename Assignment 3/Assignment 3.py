
# PLEASE MODIFY PATH TO DIRECTORY PATH IN LINE 16, 19, 21

# Part 1 took an estimated 8.3 minutes in Colab as number of rows is around 50000 !

# Part 1 

import pandas as pd
import re
import nltk
import itertools

tokens = []
count = 0

nltk.download('punkt')

df = pd.read_csv("D:/UCSC/UCSC/NLP 220 - Data Collection, Wrangling and Crowdsourcing/Assignments/Assignment 3/semeval-2017-train 2.csv")
df.dropna(axis = 1)

df1 = pd.read_csv(r"D:/UCSC/UCSC/NLP 220 - Data Collection, Wrangling and Crowdsourcing/Assignments/Assignment 3/slang.txt",sep="=")
df1.to_csv("slang.csv", header=['Slang', 'Meaning'])
df1 = pd.read_csv(r"D:/UCSC/UCSC/NLP 220 - Data Collection, Wrangling and Crowdsourcing/Assignments/Assignment 3/slang.csv")

df2 = pd.DataFrame(columns=['original tweet', 'slang word -> replacement word'])
df_sent = pd.DataFrame(columns=['original tweet', 'Sentiment Label'])

for i in range(0,100): # Replace line with "for i in range(0,500):"

  # Remove URLS
  df['label\ttext'][i] = re.sub(r'https?:\/\/.\S+', "",  df['label\ttext'][i])

  # Remove Hashtag
  df['label\ttext'][i] = re.sub(r'#', '', df['label\ttext'][i]) 

  # Remove chars that appear more than 3 times 
  df['label\ttext'][i] = ''.join(''.join(s)[:3] for _, s in itertools.groupby(df['label\ttext'][i])) 

  # Remove tokens which starts with @
  df['label\ttext'][i] = re.sub(r'@', '', df['label\ttext'][i])

  tokens =  nltk.word_tokenize(df['label\ttext'][i])

  # Remove sentences which have less than 4 tokens
  if len(tokens) < 5:
    del df['label\ttext'][i]
    continue

  # Identify the slang words using a slang dictionary and output in a CSV
  tokens = [i for i in tokens]
  senti = tokens[0]
  df.at[i, 'Sentiment'] = senti
  tokens = [tokens[i] for i in range(0,len(tokens)) if i>0] # Remove 'if i>0' if you dont want to remove the sentiment
  count +=1
  print(count, tokens)
  for token in tokens:
    for x in range(0,len(df1['Slang'])):
      if token.upper() == df1['Slang'][x]:
        index = list(df1['Slang']).index(token.upper())
        #print([df['label\ttext'][i], df1['Slang'][index], df1['Meaning'][index]])
        df_length = len(df2)
        df2.loc[df_length] = [ " ".join(tokens), df1['Slang'][index] + '->' + df1['Meaning'][index] ]
        df_sent.loc[df_length] = [ " ".join(tokens), df['Sentiment'][i] ]

df2.to_csv("Extracted Slang.csv")

#   Part 2

import spacy
from nltk.corpus import wordnet
from itertools import chain
import random

nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')

# For each text, apply data augmentation to generate variations (max 5 variations per text)
df3 = pd.DataFrame(columns=['original text',  'augmentation1', 'augmentation2', 'augmentation3', 'augmentation4', 'augmentation5'])

for i in df2['original tweet']:

  augment1, augment2, augment3, augment4, augment5 = '','','','',''
  doc = nlp(i)

  augment1, augment2, augment3, augment4, augment5 = i,i,i,i,i
  for token in doc:
        
    synonyms = []
    if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
      
      # Replace nouns and verbs with their synonyms. Use wordnet for getting synonyms
      for syn in wordnet.synsets(str(token)):
        synonyms.append(syn.lemma_names())

      unique_synonym = list(set(list(chain.from_iterable(synonyms))))
      
      if len(unique_synonym) == 0:
        augment1 = 'None'
        augment2 = 'None'
        augment3 = 'None'
        augment4 = 'None'
        augment5 = 'None'
      else:
        augment1 = augment1.replace(str(token),random.choice(unique_synonym))
        augment2 = augment2.replace(str(token),random.choice(unique_synonym))
        augment3 = augment3.replace(str(token),random.choice(unique_synonym))
        augment4 = augment4.replace(str(token),random.choice(unique_synonym))
        augment5 = augment5.replace(str(token),random.choice(unique_synonym))

  df3_length = len(df3)
  df3.loc[df3_length] = [ i, augment1, augment2, augment3, augment4, augment5 ]

# Output the original text and augmented text in a csv file
df3.to_csv('Augmented Sentences.csv')



#      Part 3


# Print the size of your augmented dataset, ratio of original/augmented set
augSize = 0

for i in df3.columns:
  for j in df3[i]:
    if j != 'None':
      augSize += 1

print("Augmented Dataset Size is:", augSize)
print("\nOriginal Dataset Size is:", len(df3['original text']))
print("\nOriginal by Augmented Ratio:", augSize/len(df3['original text']))

# df3.head()

df4 = df3.copy(deep=True)

df4['Sentiment Label'] = df_sent['Sentiment Label']

df4['Sentiment Label'] = df4['Sentiment Label'].astype(int)

# Label distribution of the augmented set
heads = ['original text',	'augmentation1',	'augmentation2',	'augmentation3',	'augmentation4',	'augmentation5']

df_neutral = list(chain.from_iterable(df4[df4['Sentiment Label'] == 0][heads].values.tolist())) 
df_pos = list(chain.from_iterable(df4[df4['Sentiment Label'] == 1][heads].values.tolist())) 
df_neg = list(chain.from_iterable(df4[df4['Sentiment Label'] == -1][heads].values.tolist())) 

df_neutral = [i for i in df_neutral if i != 'None']
df_pos = [i for i in df_pos if i != 'None']
df_neg = [i for i in df_neg if i != 'None']


label_dist = {-1:len(df_neg), 0:len(df_neutral), 1:len(df_pos)}

print("Label Distribution of Augmented Dataset:",label_dist)



"""Generate n-gram (unigram and bigram, trigram) for the augmented dataset. Print them to console"""

import nltk
from itertools import chain

nltk.download('punkt')

def basic_clean(text):
  return [word for word in nltk.word_tokenize(text)]

# augList = ['augmentation1', 'augmentation2', 'augmentation3', 'augmentation4', 'augmentation5']

augmentedWords = []

for i in heads:
  augmentedWords.append(basic_clean((''.join(str(df3[i].tolist()))).replace(',','').replace(']','').replace('[','').replace("'","").replace('.','').replace('!','').replace('None','')))
augmentedWords = list(chain.from_iterable(augmentedWords))
print("Augmented Corpus Words: ", augmentedWords)

originalWords = []
originalWords.append(basic_clean((''.join(str(df3['original text'].tolist())).replace(',','').replace(']','').replace('[','').replace("'","").replace('.','').replace('!','').replace('None',''))))
originalWords = list(chain.from_iterable(originalWords))
print("Original Corpus Words: ", originalWords)

print("Unigrams from the augmented dataset:\n", augmentedWords)

bigrams = (pd.Series(nltk.ngrams(augmentedWords, 2)).value_counts())
print("\nBigrams from the augmented dataset:\n", bigrams.axes[0].tolist())

trigrams = (pd.Series(nltk.ngrams(augmentedWords, 3)).value_counts())
print("\nTrigrams from the augmented dataset:\n", trigrams.axes[0].tolist())


#      Part 4

#Now compute rank/frequency profile of words of the original corpus and augmented corpus.


# Frequency Distribution
augmentedFreq_dist = nltk.FreqDist(augmentedWords)
print("Augmented Corpus Frequency Distribution:",dict(augmentedFreq_dist))

originalFreq_dist = nltk.FreqDist(originalWords)
print("Original Corpus Frequency Distribution:",dict(originalFreq_dist))

# Rank

augmentedRank = {key: rank for rank, key in enumerate(sorted(augmentedFreq_dist, key=augmentedFreq_dist.get, reverse=True), 1)}
originalRank = {key: rank for rank, key in enumerate(sorted(originalFreq_dist, key=originalFreq_dist.get, reverse=True), 1)}

print("Augmented Corpus Rank Distribution:", augmentedRank)
print("Original Corpus Rank Distribution:", originalRank)

# Rank:Frequency Profile

augmentedRankFreqProf = {augmentedRank[i]:augmentedFreq_dist[i] for i in augmentedFreq_dist}
originalRankFreqProf = {originalRank[i]:originalFreq_dist[i] for i in originalFreq_dist}

print("Augmented Corpus Rank:Frequency Profile:", augmentedRankFreqProf)
print("Original Corpus Rank:Frequency Profile:", originalRankFreqProf)

"""Output the percentage of corpus size made up by the top-10 words for both original and augmented corpus. """

augTop10Count = 0
for i in augmentedFreq_dist.most_common(10):
  augTop10Count += i[1]

print("Augmented Corpus Top 10 Words Count", augTop10Count)
print("Percentage of Top 10 Words in Augmmented Corpus:", (augTop10Count / len(augmentedWords)*100))

origTop10Count = 0
for i in originalFreq_dist.most_common(10):
  origTop10Count += i[1]

print("Original Corpus Top 10 Words Count", origTop10Count)
print("Percentage of Top 10 Words in Original Corpus:", (origTop10Count / len(originalWords)*100))

#     Part - 5

# Find top-10 bi-grams and tri-grams from positive sentiment samples and top-10 bi-grams and tri-grams from negative sentiment samples using NLTK’s significant collocation approach.


pos_words = [ i.split() for i in df_pos]
pos_words = list(chain.from_iterable(pos_words))
print(pos_words)
neg_words = [ i.split() for i in df_neg]
neg_words = list(chain.from_iterable(neg_words))
print(neg_words)

# PMI to select significant collocations

from nltk.collocations import *
nltk.download('genesis')

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

biPosWords_finder = BigramCollocationFinder.from_words(pos_words)
biNegWords_finder = BigramCollocationFinder.from_words(neg_words)
triPosWords_finder = TrigramCollocationFinder.from_words(pos_words)
triNegWords_finder = TrigramCollocationFinder.from_words(neg_words)

print("\nTop-10 Bi-grams from positive sentiment samples",biPosWords_finder.nbest(bigram_measures.pmi, 10))
print("\nTop-10 Bi-gramsfrom Negative sentiment samples",biNegWords_finder.nbest(bigram_measures.pmi, 10))
print("\nTop-10 Tri-grams from positive sentiment samples",triPosWords_finder.nbest(trigram_measures.pmi, 10))
print("\nTop-10 Tri-grams from Negative sentiment samples",triNegWords_finder.nbest(trigram_measures.pmi, 10))

#  Maximum likelihood to select significant collocations 

print("\nTop-10 Bi-grams from positive sentiment samples",biPosWords_finder.nbest(bigram_measures.likelihood_ratio, 10))
print("\nTop-10 Bi-grams and tri-grams from Negative sentiment samples",biNegWords_finder.nbest(bigram_measures.likelihood_ratio, 10))
print("\nTop-10 Tri-grams from positive sentiment samples",triPosWords_finder.nbest(trigram_measures.likelihood_ratio, 10))
print("\nTop-10 Tri-grams and tri-grams from Negative sentiment samples",triNegWords_finder.nbest(trigram_measures.likelihood_ratio, 10))

"""## Part - 6"""

dataset_words = originalWords + augmentedWords
print(dataset_words)

all_sents2D = []
for i in df3.columns:
  all_sents2D.append(df3[i].tolist())
all_sents = list(chain.from_iterable(all_sents2D))
print(all_sents)
print(all_sents2D)

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

sents = []
tfidf_enc = TfidfVectorizer()
tfidf = tfidf_enc.fit_transform(all_sents)
pd.set_option('display.max_colwidth',100)
df9 = pd.DataFrame(tfidf.toarray())
df9.columns = tfidf_enc.get_feature_names()
df9.index = all_sents
print(df9.head())

# Label Encoding

from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


label_enc = preprocessing.LabelEncoder()
labels = label_enc.fit_transform(dataset_words)
my_encodings = dict(zip(label_enc.classes_, range(len(label_enc.classes_))))
print(my_encodings)

# One Hot Encoding
import numpy as np
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dff = pd.DataFrame(all_sents)
ohe = preprocessing.OneHotEncoder(sparse=False)
sentences_ohe = ohe.fit_transform(dff[[0]])
df11 = pd.DataFrame(sentences_ohe)
df11.index = all_sents
df11.columns = ohe.categories_
print(df11.head())

#Alternative method to create One Hot Encoding
print(pd.get_dummies(all_sents))



"""THANK YOU !"""

