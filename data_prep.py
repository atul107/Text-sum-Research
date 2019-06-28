# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:05:02 2019

@author: atk
"""
from nltk.tokenize import sent_tokenize
import json


train_article = []
train_title = []
"""
title_tokenized
permalink
title
url
num_comments
tldr
created_utc
trimmed_title_tokenized
id
selftext_html
score
upvote_ratio
selftext
selftext_without_tldr_tokenized
ups
selftext_without_tldr
"""

with open('tifu_all_tokenized_and_filtered.json', 'r') as fp:
    for line in fp:
     current_dict = json.loads(line)
     data =[]
     data.append(sent_tokenize(current_dict['selftext_without_tldr']))
     
     for text in data:
         s = " ".join(text)
         
     train_article.append(s)     
     train_title.append(current_dict['trimmed_title'])
     


with open("train.article.txt", "wb") as out:
    for item in train_article:
        out.write("%s\n" % item.encode('utf-8'))

with open('train.title.txt', 'wb') as out:
    for item in train_title:
        out.write("%s\n" %item.encode('utf-8'))      