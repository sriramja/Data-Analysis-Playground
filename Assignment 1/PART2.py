from re import A
import nltk
from nltk.corpus import gutenberg, reuters
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

# Import Reuters corpus (all the documents in the corpus) from NLTK.

nltk.download('reuters')
file = nltk.corpus.reuters.fileids()
words = nltk.corpus.reuters.words()
doc = nltk.corpus.reuters.raw()
wordtokens = word_tokenize(doc)
#print(len(words))

# Removing Stopwords and Non-alphanumeric tokens
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('I')
stop_words.add('Mr')
tokensWithoutStopwords = [word for word in wordtokens if word not in stop_words]
tokensWithoutStopwordsAlNum = [e for e in tokensWithoutStopwords if e.isalnum()]

# Find frequency distribution of the words.
frequency = dict(Counter(tokensWithoutStopwordsAlNum).most_common(20))
print("The frequencies are:", frequency)

fd = FreqDist(tokensWithoutStopwordsAlNum)
print(fd)
