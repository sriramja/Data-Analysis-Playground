import nltk
from nltk.corpus import reuters
nltk.download('reuters')
categories = reuters.categories()
print(categories[:5])