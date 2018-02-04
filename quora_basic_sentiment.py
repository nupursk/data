#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:39:32 2018

@author: nupur
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('quora_duplicate_questions.tsv', sep='\t', index_col=[0,1,2])


from textblob import TextBlob
import string 

text1 = []
polarity1 = []
subjectivity1 = []
phrases1 = []

text2 = []
polarity2 = []
subjectivity2 = []
phrases2 = []

for i in df.question1:
    text1.append(i)
for i in df.question2:
    text2.append(i)

printable = set(string.printable)

for i in range(len(text1)):
    if type(text1[i])==float:
        text1[i] = 'nan'
    text1[i] = filter(lambda x: x in printable, text1[i])
#   print(text[i])
    blob1 = TextBlob(text1[i])
    polarity1.append(blob1.polarity)
    subjectivity1.append(blob1.subjectivity)
    phrases1.append(blob1.noun_phrases)
    
for i in range(len(text2)):
    if type(text2[i])==float:
        text2[i] = 'nan'
    text2[i] = filter(lambda x: x in printable, text2[i])
#   print(text[i])
    blob2 = TextBlob(text2[i])
    polarity2.append(blob2.polarity)
    subjectivity2.append(blob2.subjectivity)
    phrases2.append(blob2.noun_phrases)
   

#for i in range(10):
#
#    text[i] = filter(lambda x: x in printable, text[i])
#    print(text[i])
#    blob = TextBlob(text[i])
#    print(blob.polarity)
#    print(blob.subjectivity)
#    print(blob.tags)     # [('The', 'DT'), ('titular', 'JJ'),('threat', 'NN'), ('of', 'IN'), ...]            
#    print(blob.noun_phrases) ## WordList(['titular threat', 'blob', 'ultimate movie monster','amoeba-like mass', ...])
#    for sentence in blob.sentences:
#        print(sentence.sentiment.polarity)

plt.scatter(polarity1,polarity2)
plt.show()

plt.scatter(subjectivity1,subjectivity2)
plt.show()

