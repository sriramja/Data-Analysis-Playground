import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize, TweetTokenizer 

nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Find the total number of Hashtags, Mentions and URLs in the corpus.

tokenizerr = TweetTokenizer(strip_handles=False, reduce_len=True)
tokens = []

for t in twitter_samples.strings("tweets.20150430-223406.json"):   
    tokens.append(tokenizerr.tokenize(t))

hashtag_count = 0
url_count = 0
mention_count = 0
hashtags = []
mentions = []

for token in range(0,len(tokens)):
    for t in range(0,len(tokens[token])):
        if tokens[token][t].startswith('http' or 'www'):
            url_count +=1  # Count the number of URLs
        if tokens[token][t].startswith('#'):
            hashtag_count += 1   # Count the number of Hashtags
            tokens[token][t].replace(tokens[token][t],"")  # After counting, REMOVE the counted Hashtag
        if tokens[token][t].startswith('@'):
            mention_count +=1 # Count the number of Mentions
            tokens[token][t].replace(tokens[token][t],"") # After counting, REMOVE the counted mention

print("\nnumber of URLs:",url_count)
print("\nnumber of Hashtags:",hashtag_count)
print("\nnumber of Mentions:",mention_count)
print("\n\nTweets after removing Hashtags and Mentions:", *tokens[token],sep = ", ")

for token in range(0,len(tokens)):
    print(*tokens[token],sep = ", ")
