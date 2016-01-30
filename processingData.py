# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:54:06 2016

@author: brian
"""
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *

from sklearn import svm
from sklearn.metrics import accuracy_score

'''
Reference: http://francescopochetti.com/scrapying-around-web/
Borrowed from link above. Converts the list to strings
'''
def unlist(element):
    return ''.join(element)

'''
Reference: http://francescopochetti.com/scrapying-around-web/
Read jason and store into DataFrame
Convert lists to strings
Convert date column to DateTime
Drop body and title since unneeded when using keywords
Remove all empty strings
'''
def read_scraped_jason(filename):
    df = pd.read_json(filename)
    
    for column in df.columns:
        df[column] = df[column].apply(unlist)
    # gets only first 10 characters of date: year/month/day
    df['date'] = df['date'].apply(lambda x: x[:10])
    df['date'] = pd.to_datetime(df['date'])
    
    # if any removes duplicate posts
    df = df.drop_duplicates(subset = ['keywords'])
    # sorts dataframe by post date
    df = df.sort_values(by='date')
 
    df = df.drop('body', 1)
    df = df.drop('title', 1)
    
    df['keywords'].replace('', np.nan, inplace=True)
    df = df.dropna()
    
    return df

'''
Start main script
'''

      
'''
Reading in jason. After this date and keyword column remains
'''
keyword_df = read_scraped_jason('busweek/items.json')

'''
Reading in Yahoo Finance csv
Code borrowed from ML for Trading on Udacity
'''
market_df = pd.read_csv('sp_500_2014.csv', index_col="Date",
                        parse_dates=True, usecols=['Date','Open','Close'],
                        na_values=['nan'])
                        
market_df = market_df.drop(pd.to_datetime('2014-12-24'))

'''
Grabbing all the unique words in the keyword_df
This will be used to determine which of the keywords
we'll use in the actual SVM algorithm
Sorting by frequency and now using only top 50
'''
word_dict = dict()
for index, row in keyword_df.iterrows():
    word_list = row.loc['keywords'].strip().split(',')
    for word in word_list:
        if not unicode.isdigit(word):
            if word_dict.has_key(word):
                word_dict[word] = word_dict[word] + 1;
            else:
                word_dict[word] = 1;
                
sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
features = sorted_words[0:50]
features_dict = {}
for i in range(len(features)):
    features_dict[features[i]] = i;

'''
the keyword dataframe contains rows for each article, not day
so I want to combine all days so I can compare the market data
easier
'''
data_set = pd.DataFrame(index=market_df.index, columns=['keywords'])

for i, row in keyword_df.iterrows():
    
    if keyword_df.loc[i, 'date'] in data_set.index:
        if pd.isnull(data_set.loc[keyword_df.loc[i, 'date']]).any():
            data_set.loc[keyword_df.loc[i, 'date'], 'keywords'] = keyword_df.loc[i, 'keywords']
        else:
            data_set.loc[keyword_df.loc[i, 'date'], 'keywords'] += keyword_df.loc[i, 'keywords']

'''
here I'm creating the actual feature dataset
the feature dataframe is 50 columns wide for the 50 keywords
being used and a row for each day the market was open. a one is entered in 
the cell if the word is present on that day.
'''
feature_df = pd.DataFrame(index=data_set.index, columns=[range(50)])
feature_df = feature_df.fillna(0)

for i, row in data_set.iterrows():
    word_list = row.loc['keywords'].strip().split(',')
    for word in word_list:
        if word in features:
            feature_df.loc[i, features_dict[word]] = 1;

'''
to make things simpler for me, I'm only using a binary value to denote
rising/falling for the market on a particular day
'''
output_df = pd.DataFrame(index=data_set.index, columns=['rising'])
for i, row in market_df.iterrows():
    if row['Close'] > row['Open']:
        output_df.loc[i, 'rising'] = 1
    else:
        output_df.loc[i, 'rising'] = 0

'''
Creating the feature/label dataframes for train/validation/test
using an 60/20/20 split approx.
'''            
features_train = feature_df.sample(n = 150)
train_ind = features_train.index

helper_df = feature_df.drop(train_ind)
features_validate = helper_df.sample(n = 50)
validate_ind = features_validate.index

features_test = helper_df.drop(validate_ind)
test_ind = features_test.index

label_train = output_df.ix[train_ind]
label_validate = output_df.ix[validate_ind]
label_test = output_df.ix[test_ind]

'''
Converting the dataframes to useful object for the SVM
algorithm
'''
features_train = features_train.as_matrix()
features_test = features_test.as_matrix()
features_validate = features_validate.as_matrix()

label_train = list(label_train.values.flatten())
label_test = list(label_test.values.flatten())
label_validate = list(label_validate.values.flatten())

'''
finding the best C value using the validation set
'''
C_values= [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

print "Trying SVM rbf now"
best_accuracy = 0

for curr_C in C_values:
                
    clf = svm.SVC(kernel = 'rbf', C = curr_C)
    clf.fit(features_train, label_train)
    
    pred = clf.predict(features_validate)
    curr_accuracy = accuracy_score(label_validate, pred)
    
    print "C = %f; accuracy: %f" %  (curr_C, curr_accuracy)
          
    if best_accuracy < curr_accuracy:
        best_accuracy = curr_accuracy
        best_C = curr_C
        
print "Best C = %f; Best Accuracy = %f" % (best_C, best_accuracy)

'''
testing on test data
'''
clf = svm.SVC(kernel = 'rbf', C = best_C)
clf.fit(features_train, label_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(label_test, pred)

print "Final test accuracy: %f" %  (accuracy)
