#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:24:40 2018

@author: nupur
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:11:19 2018

@author: nupur
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import requests

#####################################################

# Tf-idf and NMF on questions
df = pd.read_csv('quora_duplicate_questions.tsv', sep='\t', index_col=[0,1,2])

#preprocessing for tokenization
df.iloc[301412,0]=df.iloc[301412,0].replace(' ','_')   # ' ' at position 21 replaced by '_'
df.iloc[367305,0]=df.iloc[367305,0].replace(' ','_')
df.iloc[105780,1]=str(df.iloc[105780,1])  #replaced nan (type=float) by str: 'nan'
df.iloc[161482,1]='Incentre of _ ABC is I. _ABC = 90_ and _ = _ is ?'
df.iloc[316559,1]='What does this mean in English? ___'
df.iloc[201841,1]=str(df.iloc[201841,1])


# Generate vocabulary for Tf-idf
#Word tokenization with NLTK
from nltk.tokenize import word_tokenize
tokens_q1 = []
tokens_q2 = []
for i in range(len(df)):
    tokens_q1.append(word_tokenize(df.iloc[i,0]))
    tokens_q2.append(word_tokenize(df.iloc[i,1]))
tokens = tokens_q1 + tokens_q2

text = tokens[:]                    # select document slice to be analyzed

#pre-processing tokens
# list of stop words: (ranksnl_oldgoogle.txt + analysis)
english_stops = ['a','about','an','and','are','as','at','be','by','can','do','did','doe','for','from','get','how','have','i','if','in','is','it','my','not','of','on','one','or','some','should','that','the','there','this','to','you','your','was','we','what','when','where','which','who','why','will','with','would','the']

#Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
# Instantiate the WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()

all_words = []
for i in range(len(text)):
    text[i] = [t.lower() for t in text[i] if t.isalpha()]       # make all words lowercase
    text[i] = [t for t in text[i] if t.isalpha()]               # retain only alphabetic words
    text[i] = [t for t in text[i] if t not in english_stops]    # remove stopwords
    text[i] = [wordnet_lemmatizer.lemmatize(t) for t in text[i]]# lemmatize
    
    for j in (text[i][0:]):
        if j not in all_words:
            all_words.append(j)


#Tf-idf on questions
#  Import TfidfVectorizer
text1 = df.iloc[i,0]
text2 = df.iloc[i,1]
text = text1+text2
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', vocabulary=all_words)
matrix = vectorizer.fit_transform(text) 

#save matrix
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
    
save_sparse_csr('tfidfmatrix', matrix)  #saved as tfidfmatrix.npz



# NMF on questions
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    
articles = load_sparse_csr('tfidfmatrix.npz')

titles000 = text

from sklearn.decomposition import NMF

model = NMF(n_components = 6)

model.fit(articles)

nmf_features000 = model.transform(articles)

df1000 = pd.DataFrame(nmf_features000,index=titles000)


##################################################

# Tf-idf and NMF on celebrity information
names = pd.read_csv('mostinfluential.csv')

wiki=[]

for i in names.Celebrity:
    #The World's Most Powerful People_Forbes
    a = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles='
    b = i
    url = a+b
# Package the request, send the request and catch the response: r 
    r = requests.get(url)

# Decode the JSON data into a dictionary: json_data 
    json_data = r.json()

    for k in json_data.keys():
        if 'query' in k:
            wiki.append(str(json_data[k])[37:-25])
            
#  Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', vocabulary=all_words)
matrix = vectorizer.fit_transform(wiki)   #text=text1+text2
#matrix.shape: (808580, 139466)  type(matrix): scipy.sparse.csr.csr_matrix


#save matrix    
save_sparse_csr('wikimat', matrix)  #saved as tfidfmatrix.npz

#load matrix
wikimat = load_sparse_csr('wikimat.npz')

titles = names.Celebrity

from sklearn.decomposition import NMF

model = NMF(n_components = 6)

model.fit(articles)

nmf_features = model.transform(wikimat)

# test
df1 = pd.DataFrame(nmf_features,index=titles)


############# cross_recommendations ###########################
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features000) #<-- questions(808580,6)
df2 = pd.DataFrame(nmf_features000[:100],index=titles000[:100]) #<-- select a subset of questions(808580,6) if needed; titles000=actual qns:808580
article = df1.iloc[16] # or: article = df1.loc['question string']
print article
print 'similarities:\n',similarities.nlargest()
# sim = similarities.sort_values(ascending=False) & print sim[:20] to get the closest 20 similarities

###### other way round:

norm_features = normalize(nmf_features)
df2 = pd.DataFrame(nmf_features,index=titles) 
article = df1000.iloc[160] #df1 = pd.DataFrame(nmf_features,index=titles) # for questions
similarities = df2.dot(article)
print article
print 'similarities:\n',similarities.nlargest()
sim = similarities.sort_values(ascending=False)

#topics of documents
components_df = pd.DataFrame(model.components_, columns=all_words)
print(components_df.shape)
component = components_df.iloc[3]  #replace 3 by 0-5 in a loop to check all components
print(component.nlargest())