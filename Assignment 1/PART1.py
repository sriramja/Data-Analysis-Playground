import nltk
from collections import Counter
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.probability import FreqDist


# Import a book from the Gutenberg Project in NLTK, and tokenize the text
nltk.download('gutenberg')
file = nltk.corpus.gutenberg.fileids()
raw = nltk.corpus.gutenberg.raw('austen-emma.txt')
words = nltk.corpus.gutenberg.words('austen-emma.txt')
ner_model = spacy.load("en_core_web_sm")

doc = ner_model(raw)
wordtokens = word_tokenize(raw)

# Removing Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('I')
stop_words.add('Mr')
tokensWithoutStopwords = [word for word in wordtokens if word not in stop_words]

print(len(words))  # Count of words before removing stopwords
print(len(tokensWithoutStopwords)) # Count of words after removing stopwords    

#Removing Non-alphanumeric tokens

tokensWithoutStopwordsAlNum = [e for e in tokensWithoutStopwords if e.isalnum()]
print(len(tokensWithoutStopwordsAlNum)) # Count of words after removing stopwords and non alphanumeric tokens


# Compute the vocabulary of the book. To do that, you will need to find the frequency distribution of tokens. 
# Save the distribution in a CSV file using the format:    token: frequency

frequency = dict(Counter(tokensWithoutStopwordsAlNum))
frequency_50 = dict(Counter(tokensWithoutStopwordsAlNum).most_common(50))
csv_file = "Frequency_Assignment.csv"

with open(csv_file, 'w') as f:
    f.write("Token,Frequency")
    for key in frequency.keys():
        f.write("%s,%s\n"%(key,frequency[key]))

# Determine the POS tags for the book’s entire text, and find the frequency distribution of the POS tags. 
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
book_pos = nltk.pos_tag(tokensWithoutStopwordsAlNum)
frequency_pos = dict(Counter(book_pos))
frequency_pos_50 = dict(Counter(book_pos).most_common(50))

lower_case = str(doc).lower()
tokens = nltk.word_tokenize(lower_case)
tags = nltk.pos_tag(tokens)
counts = dict(Counter( tag for word,  tag in tags))
print("\nFrequency distrubution of POS tags:",counts)


# Cumulative frequency distribution of the most frequent tokens
fig = plt.figure(figsize = (10,4))
plt.gcf().subplots_adjust(bottom=0.15)
freq = nltk.FreqDist(frequency_50)
freq.plot(50,cumulative=True)
plt.show()
fig.savefig('Cumulative frequency distribution.png')

# simple frequency distribution of the POS tags
plt.rcParams["figure.figsize"] = (30,3)
plt.bar(range(len(counts.keys())), list(counts.values()), align='center')
plt.xticks(range(len(counts.keys())), list(counts.keys()))
plt.savefig('POS simple frequency distribution.png')
plt.show()


# Use a corpus for Names or Open source tool (e.g., Spacy) to find the person names in the book 
# and output the most frequent name. 

names = []
for ent in doc.ents:
    if ent.label_ == 'PERSON':
        names.append(ent.text)

#print("Names", names)
Names_freq = Counter(names).most_common(1)
print("\nMost Common name is:",dict(Names_freq))
