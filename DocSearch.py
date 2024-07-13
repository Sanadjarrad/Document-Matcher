import nltk
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import snowball
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import collections
import pandas as pd
import math
import numpy as np
import itertools
import bisect


# Reading the file
def file_read(file_name):
    with open(file_name) as c1:
        corpus = c1.readlines()
    return corpus

# Counting each unique word from the file
def word_counter(file_name):
    with open('docs.txt') as f3:
        count = 0
        file = open(file_name, 'r')
        read_words = file.read()
        words = set(read_words.split())
        for word in words:
            count += 1
    return count

corpus = file_read('docs.txt')


# List to store the words that were in the file
corpus_words = []


# Tokenize a paragraph into sentences and each sentence in to a word
for word in corpus:
    for sent in sent_tokenize(word):
        wordsTokens = word_tokenize(sent)
        corpus_words += wordsTokens

# Eleminating duplicates
lower_case_words = set([ x.lower() for x in corpus_words ])

# Removing stopwords // not sure if needed 
remove_stop = set(stopwords.words('english'))

# Using set difference to eliminate stopwords from our words
stopfree_words = lower_case_words - remove_stop

# Reducing words to their shortened version to improve accuracy using snowball stemmer
snow_stemmer = snowball.SnowballStemmer('english')
stemmed_words = set([snow_stemmer.stem(x) for x in stopfree_words])


# Index is a map of word and the documents it is found in // doc 0 = line 1
# Building the dictionary // Using defaultdict object to return default values if word is not found
inverted_index = defaultdict(set)


# Maintain the reference to the document by its index in the corpus list
for docid, word in enumerate(corpus):
    for sent in sent_tokenize(word):
        for word in word_tokenize(sent):
            word_lowercase = word.lower()
            if word_lowercase not in remove_stop:
                word_stem = snow_stemmer.stem(word_lowercase)
                # We add the document to the set againt the word in our index
                inverted_index[word_stem].add(docid)


# Reading the queries file and extracting each query
def query_reader(query_file):
    with open(query_file) as f1:
        queries = f1.readlines()
    return queries


def word_search(query):
    matched_documents = set()
    for word in word_tokenize(query):
        word_lower = word.lower()
        if word_lower not in remove_stop:
            word_stem = snow_stemmer.stem(word_lower)
            matches = inverted_index.get(word_stem)
            if matches:
                # |= is set union
                matched_documents |= matches
                # set_to_dict = dict.fromkeys(matched_documents, 0)
    return matched_documents
    
    
# Function to calculate cosine similarity
def cos_angle(query, docs):
    norm_query = np.linalg.norm(query)
    norm_docs = np.linalg.norm(docs)
    norm_2 = norm_query * norm_docs
    if norm_2 == 0:
        return 90
    cos_theta = np.dot(query, docs) / (norm_2)
    theta = math.degrees(math.acos(cos_theta))
    return theta
    

# Vectorizing the Corpus
Count_vec = CountVectorizer(min_df=0., max_df=1.)

Corpus_matrix = Count_vec.fit_transform(corpus)
Corpus_array = Corpus_matrix.toarray()

# Vectorizing the Query
Query_matrix = Count_vec.fit_transform(query_reader('queries.txt'))
# Resize the query matrix to make it size of corpus matrix
Query_matrix.resize(Corpus_array.shape)
Query_array = Query_matrix.toarray()



print(cos_angle(Query_array, Corpus_array))



            
print("Words in dictionary: {}\n".format(word_counter(file_name="docs.txt")))
# Docs are starting from 0 
for line in query_reader('queries.txt'):
    print("Query: {}\nRelevant documents: {}\n".format(line, word_search(query=line)))

